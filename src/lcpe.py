import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import logging, time, pickle
from cpe import SqMaskGen
from skimage.segmentation import slic,mark_boundaries
from reports import report_duration
from torch.optim.lr_scheduler import StepLR

tqdm = lambda x: x

class LoadMaskGen:
    def __init__(self, path):
        with open(path, "rb") as mf:
            self.masks = pickle.load(mf).all_masks
            self.idx = 0
        
    def gen_masks(self, batch_size):
        if self.idx + batch_size > self.masks.shape[0]:
            raise Exception("ran out of masks")
        rv = self.masks[self.idx:self.idx+batch_size]
        self.idx += batch_size
        return rv


class SegMaskGen:

    def __init__(self, inp, n_segments):
        base = inp[0].cpu().numpy().transpose(1,2,0)
        #print(base.shape)
        #n_segments = n_segments #base.shape[0] * base.shape[1] / (segsize * segsize)
        self.segments = torch.tensor(slic(base,n_segments=n_segments,compactness=10,sigma=1), dtype=torch.int32)        
        #print(inp.shape, self.segments.shape)
        self.nelm = torch.unique(self.segments).numel()
        self.mshape = inp.shape
    
    def gen_masks(self, nmasks):
        return (self.gen_masks_cont(nmasks) < 0.5)

    def gen_masks_cont(self, nmasks):        
        #print("Generating segments mask")
        step = self.nelm
        nelm = step * nmasks
        rnd = torch.rand(2+nelm)
        stt = []
        for idx in range(nmasks):
            wseg = self.segments + step * idx
            stt.append(wseg)
        parts = torch.stack(stt)
        #print(self.nelm, rnd.shape, self.segments.shape, len(self.segments.unique()), len(parts.unique()))
        return rnd[parts.view(-1)].view(parts.shape)


class MaskedRespGen:
    def __init__(self, segsize=48, ishape = (224,224),
                 mgen=None, baseline=None):

        self.segsize = segsize
        self.ishape = ishape
        if mgen is None:
            self.mgen = SqMaskGen(segsize, mshape=ishape)
        else:
            self.mgen = mgen

        if baseline is None:
            self.baseline = torch.zeros(ishape)
        else:
            self.baseline = baseline

        self.all_masks = []
        self.all_pred = []

    def gen_(self, model, inp, itr=125, batch_size=32):
        
        h = self.ishape[0]
        w = self.ishape[1]
        
        baseline = self.baseline.to(inp.device)
        
        for idx in tqdm(range(itr)):            
            masks = self.mgen.gen_masks(batch_size)
            dmasks = masks.to(inp.device).float()

            pert_inp = inp * dmasks.unsqueeze(1) + baseline * (1.0-dmasks.unsqueeze(1))
            out = model(pert_inp) ## CHNG
            mout = out.unsqueeze(-1).unsqueeze(-1)
            
            self.all_masks.append(masks.cpu())
            self.all_pred.append(mout.cpu())


    def gen(self, model, inp, nmasks, batch_size=32, **kwargs):        
        with torch.no_grad():
            self.gen_(model=model, inp=inp, itr=nmasks//batch_size, batch_size=batch_size, **kwargs)
            if nmasks % batch_size:
                self.gen_(model=model, inp=inp, itr=1, batch_size=nmasks % batch_size, **kwargs)



class TotalVariationLoss(nn.Module):
    def __init__(self, beta=2):
        super(TotalVariationLoss, self).__init__()
        self.beta = beta

    def forward(self, x):
        # Calculate the variation in the x and y directions
        x_diff = torch.abs(x[ :, :-1] - x[ :, 1:]).pow(self.beta)
        y_diff = torch.abs(x[ :-1, :] - x[ 1:, :]).pow(self.beta)
        # Sum the variations to get the total variation loss
        loss = torch.mean(x_diff) + torch.mean(y_diff)
        return loss

class MaskedExplanationSum(nn.Module):
    def __init__(self, H=224, W=224, initial_value=None,):
        super(MaskedExplanationSum, self).__init__()
        # Initialize explanation with given initial value or zeros
        if initial_value is not None:
            self.explanation = nn.Parameter(initial_value)
        else:
            self.explanation = nn.Parameter(torch.zeros(H, W))

    def forward(self, x):
        y =  (x * self.explanation).flatten(start_dim=1).sum(dim=1)
        return y

    def normalize(self, score):
        # Normalize explanation so that its sum equals the provided score
        with torch.no_grad():  # No gradient update needed for normalization
            self.explanation *= score / self.explanation.sum()

# Define the training function

def normalize_explanation(explanation, score, c_norm, c_activation):
    if c_activation == "sigmoid":        
        explanation = torch.sigmoid(explanation)        
    elif c_activation == "tanh":        
        explanation = torch.tanh(explanation)
    elif c_activation:
        assert False, f"unexpected activation {c_activation}"

    sig = explanation
    if c_norm:
        explanation = explanation * score / explanation.sum()
    return explanation, sig


def qmet(smdl, inp, sal, steps):
    with torch.no_grad():    
        bars = sal.quantile(steps).unsqueeze(1).unsqueeze(1)
        del_masks = (sal.unsqueeze(0) < bars)
        ins_masks = (sal.unsqueeze(0) > bars)        
        del_pred = smdl(del_masks.unsqueeze(1) * inp)
        ins_pred = smdl(ins_masks.unsqueeze(1) * inp)
        del_auc = ((del_pred[1:]+del_pred[0:-1])*0.5).mean() 
        ins_auc = ((ins_pred[1:]+ins_pred[0:-1])*0.5).mean() 
        return del_auc.cpu().tolist(), ins_auc.cpu().tolist()


def optimize_explanation_i(
        fmdl, inp, mexp, data, targets, epochs=10, 
        lr=0.001, lr_step=0, lr_step_decay=0,
        score=1.0, 
        c_mask_completeness=1.0, c_smoothness=0.1, c_completeness=0.0, c_selfness=0.0,        
        c_magnitude=0,
        c_tv=0, avg_kernel_size=(5,5),
        c_model=0,
        c_activation=None,
        c_norm=False,
        renorm=False, baseline=None, 
        callback=None, 
        select_from=None, select_freq=10, select_del=0.5,
        start_epoch=0,
        c_opt="Adam"
        ):
    mse = nn.MSELoss()  # Mean Squared Error loss
    bce = nn.BCELoss(reduction="mean")
    tv = TotalVariationLoss()

    if baseline is None:
        assert False ## no default
        baseline = torch.zeros(inp.shape).to(inp.device)

    #print(list(model.parameters()))
    logging.debug(f"### lr={lr}; c_completeness={c_completeness}; c_tv={c_tv}; c_smoothness={c_smoothness}; avg_kernel_size={avg_kernel_size}")
    print(f"### lr={lr}; c_completeness={c_completeness}; c_tv={c_tv}; c_smoothness={c_smoothness}; c_magnitude={c_magnitude}; avg_kernel_size={avg_kernel_size}; c_norm={c_norm}; c_activation={c_activation}; c_model={c_model}; c_opt={c_opt};")

    print("###", dict(epochs=epochs, lr=lr, score=score, 
        c_mask_completeness=c_mask_completeness, c_smoothness=c_smoothness, c_completeness=c_completeness, c_selfness=c_selfness,
        c_magnitude=c_magnitude,
        c_tv=c_tv, avg_kernel_size=avg_kernel_size, c_model=c_model,
        c_activation=c_activation, c_norm=c_norm, renorm=renorm,
        select_from=select_from, select_freq=select_freq, select_del=select_del))
    
    if c_opt=="Adam":
        optimizer = optim.Adam(mexp.parameters(), lr=lr)
    elif c_opt=="AdamW":
        optimizer = optim.AdamW(mexp.parameters(), lr=lr)
    elif c_opt == "SGD":
        optimizer = optim.SGD(mexp.parameters(), lr=lr)
    else:
        assert False, f"unexpected optimizer {c_opt}"

    scheduler = None
    if lr_step:
        scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_step_decay)

    #if not c_activation:
    #    mexp.normalize(score)

    mexp.train()
    avg_kernel = torch.ones((1,) + avg_kernel_size).to(data.device)
    avg_kernel = avg_kernel / avg_kernel.numel()

    mweights = data.flatten(start_dim=1).sum(dim=1) * 2
    #mweights = mexp.explanation.numel()

    metric_steps = torch.tensor([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]).to(inp.device)
    selection = None

    #print("$$$", mweights.shape)
    for epoch in range(start_epoch, epochs):
                
        # Forward pass
        optimizer.zero_grad()
        output = mexp(data)
        explanation, sig = normalize_explanation(mexp.explanation, score, c_norm, c_activation)
        
        comp_loss = (((output - targets) ** 2) / (mweights * explanation.numel())).mean()        
        
        #comp_loss = (((output / mweights) - (targets / mweights)) ** 2).mean()
                
        #comp_loss = mse(output/explanation.numel(), targets/explanation.numel())


        if c_completeness != 0:            
            explanation_sum = mexp.explanation.sum()
            explanation_loss = mse(explanation_sum/explanation.numel(), score/ explanation.numel())             
        else:
            explanation_loss = 0
        
        conv_loss = 0
        if c_smoothness != 0:                
            sexp = F.conv2d(explanation.unsqueeze(0), avg_kernel.unsqueeze(0),padding="same").squeeze()
            conv_loss = mse(explanation, sexp)            
        else:
            conv_loss = 0

        if c_tv != 0:
            tv_loss = tv(explanation)
        else:
            tv_loss = 0

        if c_model:
            if c_activation == "sigmoid":
                explanation_mask = sig
            else:
                explanation_mask = (explanation - explanation.min()) / (explanation.max() - explanation.min())
            masked_inp = explanation_mask * inp + (1-explanation_mask) * baseline            
            prob = fmdl(masked_inp)            
            model_loss = -torch.log(prob)
        else:
            model_loss = 0

        if c_magnitude != 0:
            magnitude_loss = (mexp.explanation.abs()).mean()
        else:
            magnitude_loss = 0

        ## tar loss
        if c_selfness != 0:
            mask_size = data.flatten(start_dim=1).sum(dim=1)
            exp_val = data * targets.unsqueeze(1).unsqueeze(1) / mask_size.unsqueeze(1).unsqueeze(1)
            act_val = data * explanation.unsqueeze(0)
            tar_loss = mse(exp_val, act_val)
        else:
            tar_loss = 0

        # Backward pass and optimization        
        total_loss = (
            c_mask_completeness * comp_loss + 
            c_smoothness * conv_loss + 
            c_completeness * explanation_loss + 
            c_tv * tv_loss +
            c_selfness * tar_loss +
            c_model * model_loss + 
            c_magnitude * magnitude_loss
            )
        
        total_loss.backward()
        if callback is not None:            
            callback(epoch=epoch, 
                    explanation=explanation.detach().clone().cpu(),
                    loss = total_loss.detach().clone().cpu(),
                    comp_loss = comp_loss.detach().clone().cpu(),
                    rexp = mexp.explanation.grad.detach().sum(),
                    grad = mexp.explanation.detach().mean())
            
        
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Normalize the explanation with the given score
        if epoch % 100 == 0 and renorm:
            mexp.normalize(score)
        
        if epoch % 10 == 0:
            pdesc = (f"Epoch {epoch+1}/{epochs} Loss={total_loss.item()}; ES={explanation.sum()}; comp_loss={comp_loss}; "
                    f"exp_loss={explanation_loss}; conv_loss={conv_loss}; tv_loss={tv_loss}; model_loss={model_loss};"
                    f"magnitude_loss={magnitude_loss}")
            logging.debug(pdesc)
            print(pdesc)

        if select_from is not None and epoch > select_from and (epoch % select_freq == 0 or epoch == epochs-1):
            dexpl = explanation.detach().clone()
            del_score, ins_score = qmet(fmdl, inp, dexpl , metric_steps)
            print(f"[{epoch}] scores: {del_score} {ins_score}")
            met_score = ins_score - select_del * del_score
            if selection is None or met_score > selection[0]:
                print("selected")
                selection = (met_score, dexpl)

    if selection:
        print("selection:", selection[0])        
        return selection[1]
    return explanation.detach().clone()

def optimize_explanation(fmdl, inp, initial_explanation, data, targets, score=1.0, 
                         epochs=0, model_epochs=0, c_model=0,
                         c_activation=None, c_norm=False,                         
                         **kwargs):
    # Initialize the model with the given initial explanation
    mexp = MaskedExplanationSum(initial_value=initial_explanation)
    mexp = mexp.to(data.device)

    if not c_activation:
        mexp.normalize(score)    
    # Train the model by passing all additional arguments through kwargs
    start_time = time.time()
    if epochs > model_epochs:
        rv = optimize_explanation_i(
            fmdl, inp, mexp, data, targets, score=score, 
            c_activation=c_activation, c_norm=c_norm, c_model=0, epochs=epochs-model_epochs, **kwargs)
    
    mid_time = time.time()
    logging.info(f"Optimization I: {mid_time - start_time}")    
    print(f"Optimization I: {mid_time -start_time}")    
    if model_epochs:
        rv = optimize_explanation_i(
            fmdl, inp, mexp, data, targets, score=score, 
            c_activation=c_activation, c_norm=c_norm,
            c_model=c_model, epochs=epochs, 
            start_epoch=(epochs-model_epochs),
            **kwargs)                
        
    end_time = time.time()    
    logging.info(f"Optimization Done: ({mid_time - start_time}) , ({end_time - start_time})")    
    print(f"Optimization Done: ({mid_time - start_time}) , ({end_time - start_time})")    
    
    #mexp.normalize(score)
    # Return the updated explanation parameter
    #explanation = normalize_explanation(mexp.explanation, score, c_norm=True, c_activation=c_activation)[0]
    return rv


class MaskedRespData:
    def __init__(self, baseline_score, label_score, added_score, all_masks, all_pred, baseline):
        self.baseline_score = baseline_score
        self.label_score = label_score
        self.added_score = added_score
        self.all_masks = all_masks
        self.all_pred = all_pred
        self.baseline = baseline

    def subset(self, nmasks):
        assert nmasks <= self.all_masks.shape[0]
        return MaskedRespData(
            baseline_score = self.baseline_score,
            label_score = self.label_score,
            added_score=self.added_score,
            all_masks=self.all_masks[0:nmasks],
            all_pred=self.all_pred[0:nmasks],
            baseline=self.baseline
        )


class ZeroBaseline:

    def __init__(self):
        pass

    def __call__(self, inp):
        return torch.zeros(inp.shape).to(inp.device)
    
    @property
    def desc(self):
        return "Zr"
    
class RandBaseline:

    def __init__(self):
        pass

    def __call__(self, inp):
        return torch.normal(0.5, 0.25, size=inp.shape).to(inp.device)    
    
    @property
    def desc(self):
        return "Rnd"

class BlurBaseline:

    def __init__(self, ksize=31, sigma=17.0):
        self.ksize = ksize
        self.sigma = sigma
        self.gaussian_blur = T.GaussianBlur(kernel_size=(ksize, ksize), sigma=sigma)        

    def __call__(self, inp):
        return self.gaussian_blur(inp)
            
    @property
    def desc(self):
        return f"Blr{self.ksize}x{self.sigma}"

class CompExpCreator:

    def __init__(self, nmasks=500, segsize=64, batch_size=32, 
                 lr = 0.05, lr_step=0, lr_step_decay=0,
                 c_mask_completeness=1.0, c_completeness=0.1, 
                 c_smoothness=0, c_selfness=0.0, c_tv=1,
                 c_magnitude=0, c_norm=False, c_activation=False,
                 c_logit = False,
                 avg_kernel_size=(5,5),
                 select_from=100, select_freq=10, select_del=0.5,
                 epochs=300, 
                 model_epochs=300, c_model=0,
                 c_opt="Adam",
                 mgen=None,
                 desc = "MComp",                 
                 baseline_gen = ZeroBaseline(),                 
                 **kwargs):
        
        assert type(segsize) == type(nmasks)
        if type(segsize) == int:
            segsize = [segsize]
            nmasks = [nmasks]
        assert len(segsize) == len(nmasks)

        self.segsize = segsize
        self.nmasks = nmasks
        self.batch_size = batch_size
        self.c_completeness = c_completeness
        self.c_smoothness = c_smoothness  
        self.c_tv = c_tv
        self.c_selfness = c_selfness
        self.c_mask_completeness = c_mask_completeness
        self.c_model = c_model
        self.c_norm = c_norm
        self.c_activation = c_activation
        self.c_logit = c_logit
        self.c_magnitude = c_magnitude
        self.c_opt = c_opt
        self.lr = lr
        self.lr_step = lr_step
        lr_step_decay = lr_step_decay
        self.avg_kernel_size = avg_kernel_size      
        self.epochs = epochs
        self.select_from = select_from
        self.select_freq = select_freq
        self.select_del = select_del
        self.model_epochs = model_epochs
        self.desc = desc
        self.mgen = mgen        
        self.baseline_gen = baseline_gen
        if self.model_epochs == 0 or self.c_model == 0:
            self.model_epochs = 0
            self.c_model = 0


    def description(self):
        if len(self.nmasks) == 1:
            desc = f"{self.desc}_{self.nmasks[0]}_{self.segsize[0]}_{self.epochs}"
        else:
            desc = f"{self.desc}_Mr_{self.epochs}"

        if self.model_epochs:
            desc += f":{self.model_epochs}"
        if self.select_from is not None:
            desc += f"b"

        if self.c_norm or self.c_activation:
            desc += "_" + ("n" * self.c_norm) + (self.c_activation[0] if self.c_activation else "")

        if self.c_logit:
            desc += "l"

        if self.c_smoothness != 0:
            desc += f"_krn{self.c_smoothness}_" + str("x").join(map(str, self.avg_kernel_size))

        if self.c_mask_completeness:
            desc += f"_msk{self.c_mask_completeness}"

        if self.c_completeness:
            desc += f"_cp{self.c_completeness}"

        if self.c_tv:
            desc += f"_tv{self.c_tv}"

        if self.c_magnitude:
            desc += f"_mgn{self.c_magnitude}"

        if self.c_model:
            desc += f"_mdl{self.c_model}"

        if self.c_selfness:
            desc += f"_sf{self.c_selfness}"

        return desc
    
    def __call__(self, me, inp, catidx, data=None):
        desc = self.description()
        sal = self.explain(me, inp, catidx, data=data)
        csal = sal.cpu().unsqueeze(0)

        return {desc : csal}
        #return {
        #    f"{self.desc}_{self.nmasks}_{self.segsize}{ksdesc}_{self.epochs}_{self.c_completeness}_{self.c_smoothness}" : sal.cpu().unsqueeze(0)
        #}


    def generate_data(self, me, inp, catidx, logit=False):
        start_time = time.time()
        baseline = self.baseline_gen(inp)
        fmdl = me.narrow_model(catidx, with_softmax=True)
        all_masks_list = []
        all_pred_list = []

        parts = list(zip(self.segsize, self.nmasks))
        for segsize, nmasks in parts:
            mgen = MaskedRespGen(segsize, mgen=self.mgen, baseline=baseline)            
            logging.debug(f"generating {nmasks} masks and responses")
            print(f"generating {nmasks} masks and responses segsize={segsize}")
            mgen.gen(fmdl, inp, nmasks, batch_size=self.batch_size)        
            all_masks_list += mgen.all_masks
            all_pred_list += mgen.all_pred
        logging.debug("Done generating masks")

        
        rfactor = inp.numel()        
        baseline_score = fmdl(baseline).detach().squeeze()
        label_score = fmdl(inp).detach().squeeze()

        if logit:
            norm = lambda x: (torch.logit(x) + torch.log((1-baseline_score)/baseline_score)) * rfactor
        else:
            norm = lambda x: (x - baseline_score) * rfactor
                
        
                
        added_score = norm(label_score)
        #print(label_score, baseline_score, delta_score)
    
        device = me.device
        all_masks = torch.concat(all_masks_list).to(device)
        
        all_pred = norm(torch.concat(all_pred_list).to(device).squeeze())
        all_pred.shape, baseline_score.shape
        
        #print("MaskGeneration,{self.segsize},{duration},")
        return MaskedRespData(
            baseline_score = baseline_score,
            label_score = label_score,
            added_score = added_score,
            all_masks = all_masks,
            all_pred = all_pred,
            baseline = baseline
        )

    def explain(self, me, inp, catidx, data=None, initial=None, callback=None):

        start_time = time.time() 
        if data is None:            
            data = self.generate_data(me, inp, catidx, logit=self.c_logit)

        start_time_expl = time.time()

        if initial is None:
            #initial = torch.rand(inp.shape[-2:]).to(inp.device)
            #initial = (torch.randn(224,224)*0.2+1).abs()
            print("setting initial")
            initial = (torch.randn(224,224)*0.2+3)

        fmdl = me.narrow_model(catidx, with_softmax=True)        
        
        sal = optimize_explanation(fmdl, inp, initial, data.all_masks, data.all_pred, score=data.added_score, 
                                   epochs=self.epochs, select_from=self.select_from, 
                                   select_freq=self.select_freq, select_del=self.select_del,
                                   model_epochs=self.model_epochs, lr=self.lr, c_opt=self.c_opt,
                                   avg_kernel_size=self.avg_kernel_size,
                                   c_completeness=self.c_completeness, c_smoothness=self.c_smoothness, 
                                   c_tv=self.c_tv, c_selfness=self.c_selfness,
                                   c_mask_completeness=self.c_mask_completeness,
                                   c_model=self.c_model,
                                   c_magnitude=self.c_magnitude,
                                   c_norm=self.c_norm, c_activation=self.c_activation,
                                   baseline=data.baseline, callback=callback)
        
        report_duration(start_time, me.arch, "LSC", f'{self.nmasks}_{self.epochs}')
        
        
        return sal


class MultiCompExpCreator:

    def __init__(self, nmasks=500, segsize=[64], batch_size=32, baselines=[ZeroBaseline()],
                 desc="MComp",
                 groups=[]):
        if type(nmasks) == int:
            self.nmasks = [nmasks]
        else:
            self.nmasks = nmasks
        self.segsize=segsize
        self.batch_size = batch_size
        self.baselines = baselines
        self.groups = groups
        self.desc = desc
        self.last_data = None

    def __call__(self, me, inp, catidx):        
        all_sals = {}
        for bgen in self.baselines:
            for segsize in self.segsize:
                dc = CompExpCreator(nmasks=max(self.nmasks), segsize=segsize, batch_size=self.batch_size,
                                    baseline_gen=bgen)
                mdata = dc.generate_data(me, inp, catidx)

                for nmasks in self.nmasks:
                    desc = self.desc + bgen.desc
                    data = mdata.subset(nmasks=nmasks)
                    self.last_data = data
                    for kwargs in self.groups:
                        group_args = dict(nmasks=nmasks, segsize=segsize, batch_size=self.batch_size, desc=desc)
                        group_args.update(kwargs)
                        algo = CompExpCreator(**group_args)
                        res = algo(me, inp, catidx, data=data)
                        all_sals.update(res)
        logging.info(f"generated: {list(all_sals.keys())}")
        return all_sals

