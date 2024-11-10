import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import logging, time, pickle
from cpe import SqMaskGen
import socket
from skimage.segmentation import slic,mark_boundaries

tqdm = lambda x: x

HOSTNAME = socket.gethostname()
def report_duration(start_time, model_name, operation, nmasks, nitr=0, with_model=False):
    duration = time.time() - start_time
    print(f"DURATION,{HOSTNAME},{model_name},{operation},{nmasks},{nitr},{int(with_model)},{duration}")


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
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        # Calculate the variation in the x and y directions
        x_diff = torch.abs(x[ :, :-1] - x[ :, 1:])
        y_diff = torch.abs(x[ :-1, :] - x[ 1:, :])
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

def optimize_explanation_i(
        fmdl, inp, mexp, data, targets, epochs=10, lr=0.001, score=1.0, 
        c_mask_completeness=1.0, c_smoothness=0.1, c_completeness=0.0, c_selfness=0.0,
        c_magnitude=0,
        c_tv=0, avg_kernel_size=(5,5),
        c_model=0,
        c_activation=None,
        c_norm=False,
        renorm=False, baseline=None, callback=None):
    mse = nn.MSELoss()  # Mean Squared Error loss
    bce = nn.BCELoss(reduction="mean")
    tv = TotalVariationLoss()

    if baseline is None:
        assert False ## no default
        baseline = torch.zeros(inp.shape).to(inp.device)

    #print(list(model.parameters()))
    logging.debug(f"### lr={lr}; c_completeness={c_completeness}; c_tv={c_tv}; c_smoothness={c_smoothness}; avg_kernel_size={avg_kernel_size}")
    print(f"### lr={lr}; c_completeness={c_completeness}; c_tv={c_tv}; c_smoothness={c_smoothness}; c_magnitude={c_magnitude}; avg_kernel_size={avg_kernel_size}; c_norm={c_norm}; c_activation={c_activation} c_model={c_model}")

    print("###", dict(epochs=epochs, lr=lr, score=score, 
        c_mask_completeness=c_mask_completeness, c_smoothness=c_smoothness, c_completeness=c_completeness, c_selfness=c_selfness,
        c_magnitude=c_magnitude,
        c_tv=c_tv, avg_kernel_size=avg_kernel_size, c_model=c_model,
        c_activation=c_activation, c_norm=c_norm, renorm=renorm))
    
    optimizer = optim.Adam(mexp.parameters(), lr=lr)

    #if not c_activation:
    #    mexp.normalize(score)

    mexp.train()
    avg_kernel = torch.ones((1,) + avg_kernel_size).to(data.device)
    avg_kernel = avg_kernel / avg_kernel.numel()

    mweights = data.flatten(start_dim=1).sum(dim=1)
    print("$$$", mweights.shape)
    for epoch in range(epochs):
                
        # Forward pass
        optimizer.zero_grad()
        output = mexp(data)
        explanation, sig = normalize_explanation(mexp.explanation, score, c_norm, c_activation)

        nweights = mweights * 2
        #comp_loss = (((output / nweights) - (targets / nweights)) ** 2).mean()
        comp_loss = mse(output/explanation.numel(), targets/explanation.numel())


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
            if c_norm or True:
                #magnitude_loss = explanation.abs().sum()
                magnitude_loss = (mexp.explanation.abs()).mean()
                #explanation_mask = explanation
                #magnitude_loss = (explanation * explanation).mean()
                #flat_mask = explanation_mask.flatten()
                #magnitude_loss = bce(flat_mask, torch.zeros(flat_mask.shape).to(flat_mask.device))
                #magnitude_loss = torch.sqrt( (explanation - score / explanation.numel()) ** 2 )
            else:
                explanation_mask = (explanation - explanation.min()) / (explanation.max() - explanation.min())            
                flat_mask = explanation_mask.flatten()
                magnitude_loss = bce(flat_mask, torch.zeros(flat_mask.shape).to(flat_mask.device)) # explanation_mask.abs().mean()
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


        # Normalize the explanation with the given score
        if epoch % 100 == 0 and renorm:
            mexp.normalize(score)
        
        if epoch % 10 == 0:
            pdesc = (f"Epoch {epoch+1}/{epochs} Loss={total_loss.item()}; ES={explanation.sum()}; comp_loss={comp_loss}; "
                    f"exp_loss={explanation_loss}; conv_loss={conv_loss}; tv_loss={tv_loss}; model_loss={model_loss};"
                    f"magnitude_loss={magnitude_loss}")
            logging.debug(pdesc)
            print(pdesc)
            


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
    optimize_explanation_i(fmdl, inp, mexp, data, targets, score=score, 
                           c_activation=c_activation, c_norm=c_norm, c_model=0, epochs=epochs-model_epochs, **kwargs)
    mid_time = time.time()
    logging.info(f"Optimization I: {mid_time - start_time}")    
    print(f"Optimization I: {mid_time -start_time}")    
    if model_epochs:
        optimize_explanation_i(fmdl, inp, mexp, data, targets, score=score, 
                               c_activation=c_activation, c_norm=c_norm,
                               c_model=c_model, epochs=model_epochs, **kwargs)
        
    end_time = time.time()    
    logging.info(f"Optimization Done: ({mid_time - start_time}) , ({end_time - start_time})")    
    print(f"Optimization Done: ({mid_time - start_time}) , ({end_time - start_time})")    
    
    #mexp.normalize(score)
    # Return the updated explanation parameter
    explanation = normalize_explanation(mexp.explanation, score, c_norm=True, c_activation=c_activation)[0]
    return explanation.detach()


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
                 lr = 0.05, c_mask_completeness=1.0, c_completeness=0.1, 
                 c_smoothness=0, c_selfness=0.0, c_tv=1,
                 c_magnitude=0, c_norm=False, c_activation=False,
                 avg_kernel_size=(5,5),
                 epochs=200, 
                 model_epochs=200, c_model=0,
                 mgen=None,
                 desc = "MComp",
                 baseline_gen = ZeroBaseline(),                 
                 **kwargs):
        
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
        self.c_magnitude = c_magnitude
        self.lr = lr
        self.avg_kernel_size = avg_kernel_size      
        self.epochs = epochs
        self.model_epochs = model_epochs
        self.desc = desc
        self.mgen = mgen
        self.baseline_gen = baseline_gen
        if self.model_epochs == 0 or self.c_model == 0:
            self.model_epochs = 0
            self.c_model = 0


    def description(self):
        desc = f"{self.desc}_{self.nmasks}_{self.segsize}_{self.epochs}"
                                
        if self.model_epochs:
            desc += f":{self.model_epochs}"

        if self.c_norm or self.c_activation:
            desc += "_" + ("n" * self.c_norm) + (self.c_activation[0] if self.c_activation else "")

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

    def generate_data(self, me, inp, catidx):        
        start_time = time.time()
        baseline = self.baseline_gen(inp)
        mgen = MaskedRespGen(self.segsize, mgen=self.mgen, baseline=baseline)
        fmdl = me.narrow_model(catidx, with_softmax=True)
        logging.debug(f"generating {self.nmasks} masks and responses")
        print(f"generating {self.nmasks} masks and responses segsize={self.segsize}")
        mgen.gen(fmdl, inp, self.nmasks, batch_size=self.batch_size,)        
        logging.debug("Done generating masks")

        rfactor = inp.numel()        
        baseline_score = fmdl(baseline).detach().squeeze() * rfactor
        label_score = fmdl(inp).detach().squeeze() * rfactor

        added_score = label_score - baseline_score
        #print(label_score, baseline_score, delta_score)
    
        device = me.device
        all_masks = torch.concat(mgen.all_masks).to(device)
        all_pred = torch.concat(mgen.all_pred).to(device).squeeze() * rfactor - baseline_score
        all_pred.shape, baseline_score.shape
        report_duration(start_time, me.arch, "MASKS", self.nmasks)        
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
         
        if data is None:            
            data = self.generate_data(me, inp, catidx)            

        start_time_expl = time.time()

        if initial is None:
            #initial = torch.rand(inp.shape[-2:]).to(inp.device)
            #initial = (torch.randn(224,224)*0.2+1).abs()
            print("setting initial")
            initial = (torch.randn(224,224)*0.2+3)

        fmdl = me.narrow_model(catidx, with_softmax=True)        
        
        sal = optimize_explanation(fmdl, inp, initial, data.all_masks, data.all_pred, score=data.added_score, 
                                   epochs=self.epochs, model_epochs=self.model_epochs, lr=self.lr, avg_kernel_size=self.avg_kernel_size,
                                   c_completeness=self.c_completeness, c_smoothness=self.c_smoothness, 
                                   c_tv=self.c_tv, c_selfness=self.c_selfness,
                                   c_mask_completeness=self.c_mask_completeness,
                                   c_model=self.c_model,
                                   c_magnitude=self.c_magnitude,
                                   c_norm=self.c_norm, c_activation=self.c_activation,
                                   baseline=data.baseline, callback=callback)
        
        report_duration(start_time_expl, me.arch, "OPT", self.nmasks, nitr=self.epochs, with_model=(self.c_model != 0))
        
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
                        algo = CompExpCreator(nmasks=nmasks, segsize=segsize, batch_size=self.batch_size, 
                                            desc=desc, **kwargs)
                        res = algo(me, inp, catidx, data=data)
                        all_sals.update(res)
        logging.info(f"generated: {list(all_sals.keys())}")
        return all_sals

