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
from collections import defaultdict
from scipy.sparse.linalg import cg, gmres, lsqr
import functools
import math

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

    def __init__(self, inp, n_segments, prob=0.5):
        base = inp[0].cpu().numpy().transpose(1,2,0)
        #print(base.shape)
        #n_segments = n_segments #base.shape[0] * base.shape[1] / (segsize * segsize)
        self.segments = torch.tensor(slic(base,n_segments=n_segments,compactness=10,sigma=1), dtype=torch.int32)        
        #print(inp.shape, self.segments.shape)
        self.nelm = torch.unique(self.segments).numel()
        self.mshape = inp.shape
        self.prob = prob
    
    def gen_masks(self, nmasks):
        return (self.gen_masks_cont(nmasks) < self.prob)

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
                 mgen=None, baseline=None, prob=0.5):

        self.segsize = segsize
        self.ishape = ishape
        if mgen is None:
            self.mgen = SqMaskGen(segsize, mshape=ishape, prob=prob)
        else:
            self.mgen = mgen

        if baseline is None:
            self.baseline = torch.zeros(ishape)
        else:
            self.baseline = baseline

        self.all_masks = []
        self.all_pred = []
        self.num_masks = 0

    def gen_(self, model, inp, itr=125, batch_size=32):
        
        h = self.ishape[0]
        w = self.ishape[1]
        
        baseline = self.baseline.to(inp.device)
        
        #print("###", inp.shape)
        for idx in tqdm(range(itr)):            
            
            masks = self.mgen.gen_masks(batch_size)
            is_valid = (masks.flatten(start_dim=1).sum(dim=1) > 0)

            if (not any(is_valid)):
                continue
            masks = masks[ is_valid ]
            dmasks = masks.to(inp.device).float()

            pert_inp = inp * dmasks.unsqueeze(1) + baseline * (1.0-dmasks.unsqueeze(1))
            out = model(pert_inp) ## CHNG
            mout = out.unsqueeze(-1).unsqueeze(-1)
            
            self.all_masks.append(masks.cpu())
            self.all_pred.append(mout.cpu())
            self.num_masks += int(is_valid.sum())


    def gen(self, model, inp, nmasks, batch_size=32, **kwargs):
        #print("GEN", nmasks, batch_size)
        with torch.no_grad():
            while self.num_masks < nmasks:                
                remaining = nmasks - self.num_masks
                #print(f"## {nmasks} - {self.num_masks} = {remaining}")
                self.gen_(model=model, inp=inp, itr=remaining//batch_size, batch_size=batch_size, **kwargs)
                if remaining % batch_size:
                    self.gen_(model=model, inp=inp, itr=1, batch_size=remaining % batch_size, **kwargs)
            #print(f"## {self.num_masks}")


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
    def __init__(self, H=224, W=224, initial_value=None, with_bias=False):
        super(MaskedExplanationSum, self).__init__()
        # Initialize explanation with given initial value or zeros
        if initial_value is not None:
            self.explanation = nn.Parameter(initial_value)
        else:
            self.explanation = nn.Parameter(torch.zeros(H, W))

        if with_bias:
            self.bias = nn.Parameter(torch.zeros(1))
            print("## with bias")
        else:
            self.bias = None

    def forward(self, x):
        y =  (x * self.explanation).flatten(start_dim=1).sum(dim=1)
        if self.bias is not None:
            y = y + self.bias
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
        c_positive=0,        
        c_compliment=False,
        c_logistic=False,
        c_activation=None,
        c_norm=False,
        renorm=False, baseline=None, 
        callback=None, 
        select_from=None, select_freq=10, select_del=0.5,
        start_epoch=0,
        c_opt="Adam"        
        ):
    mse = nn.MSELoss()  # Mean Squared Error loss
    lbce = nn.BCEWithLogitsLoss(reduction='none') 
    tv = TotalVariationLoss()

    if baseline is None:
        assert False ## no default
        baseline = torch.zeros(inp.shape).to(inp.device)

    
    logging.debug(f"### lr={lr}; c_completeness={c_completeness}; c_tv={c_tv}; c_smoothness={c_smoothness}; avg_kernel_size={avg_kernel_size}")
    print(f"## lr={lr}; c_logistic={c_logistic}; c_completeness={c_completeness}; c_tv={c_tv}; c_smoothness={c_smoothness}; c_positive={c_positive}, c_magnitude={c_magnitude}; avg_kernel_size={avg_kernel_size}; c_norm={c_norm}; c_activation={c_activation}; c_model={c_model}; c_opt={c_opt};")

    print("###", dict(epochs=epochs, lr=lr, score=score, 
        c_mask_completeness=c_mask_completeness, c_smoothness=c_smoothness, c_completeness=c_completeness, c_selfness=c_selfness,
        c_magnitude=c_magnitude, c_positive=c_positive,
        c_tv=c_tv, avg_kernel_size=avg_kernel_size, c_model=c_model,
        c_activation=c_activation, c_norm=c_norm, renorm=renorm,
        select_from=select_from, select_freq=select_freq, select_del=select_del))

    #assert (not c_logistic)

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
    compliment_mweights = (1-data*1.0).flatten(start_dim=1).sum(dim=1) * 2
    #mweights = mexp.explanation.numel()

    metric_steps = torch.tensor([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]).to(inp.device)
    selection = None

    #print("$$$", mweights.shape)
    for epoch in range(start_epoch, epochs):
                
        # Forward pass
        optimizer.zero_grad()
        output = mexp(data)
        explanation, sig = normalize_explanation(mexp.explanation, score, c_norm, c_activation)
        
        if c_logistic:
            cl_weights = 1 / data.flatten(start_dim=1).sum(dim=1)
            comp_loss = (lbce(output, targets) * cl_weights).mean()            
            assert comp_loss >= 0
        else:
            comp_loss = (((output - targets) ** 2) / (mweights * explanation.numel())).mean()        
        
        if c_compliment:
            compliment_targets = score - targets
            compliment_output = mexp(1 - data*1.0)
            compliment_comp_loss = (((compliment_output - compliment_targets) ** 2) / (compliment_mweights * explanation.numel())).mean()        
            comp_loss += compliment_comp_loss

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
            magnitude_loss = mexp.explanation.abs().mean()
        else:
            magnitude_loss = 0

        
        if c_positive != 0:
            positive_loss = ((mexp.explanation < 0) * mexp.explanation.abs()).mean() 
        else:
            positive_loss = 0

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
            c_magnitude * magnitude_loss + 
            c_positive * positive_loss
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
        
        if epoch % 100 == 0:
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
                         c_activation=None, c_norm=False, c_logistic=False,
                         **kwargs):
    # Initialize the model with the given initial explanation
    shape = inp.shape[-2:]    
    #assert (not c_logistic)
    mexp = MaskedExplanationSum(initial_value=initial_explanation, H=shape[0], W=shape[1], with_bias=c_logistic)
    mexp = mexp.to(data.device)

    if not c_activation:
        mexp.normalize(score)    
    # Train the model by passing all additional arguments through kwargs
    start_time = time.time()
    if epochs > model_epochs:
        rv = optimize_explanation_i(
            fmdl, inp, mexp, data, targets, score=score, 
            c_activation=c_activation, c_norm=c_norm, c_logistic=c_logistic, 
            c_model=0, epochs=epochs-model_epochs, **kwargs)
    
    mid_time = time.time()
    logging.info(f"Optimization I: {mid_time - start_time}")    
    print(f"Optimization I: {mid_time -start_time}")    
    if model_epochs:
        rv = optimize_explanation_i(
            fmdl, inp, mexp, data, targets, score=score, 
            c_activation=c_activation, c_norm=c_norm,
            c_model=c_model, epochs=epochs, 
            start_epoch=(epochs-model_epochs),
            c_logistic=c_logistic,
            **kwargs)                
        
    end_time = time.time()    
    logging.info(f"Optimization Done: ({mid_time - start_time}) , ({end_time - start_time})")    
    print(f"Optimization Done: ({mid_time - start_time}) , ({end_time - start_time})")    
    
    #mexp.normalize(score)
    # Return the updated explanation parameter
    #explanation = normalize_explanation(mexp.explanation, score, c_norm=True, c_activation=c_activation)[0]
    return rv


class MaskedRespData:
    def __init__(self, baseline_score, label_score, added_score, all_masks, all_pred, baseline, all_pred_raw=None):
        self.baseline_score = baseline_score
        self.label_score = label_score
        self.added_score = added_score
        self.all_masks = all_masks
        self.all_pred = all_pred
        self.baseline = baseline
        self.all_pred_raw = all_pred_raw

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

    @staticmethod
    def join(data):
        return MaskedRespData(
            baseline_score = data[0].baseline_score,
            label_score = data[0].label_score,
            added_score=data[0].added_score,
            all_masks=torch.concat([x.all_masks for x in data]),
            all_pred=torch.concat([x.all_pred for x in data]),
            baseline=data[0].baseline
        )
    
    def to(self, device):
        return MaskedRespData(
            baseline_score = self.baseline_score.to(device),
            label_score = self.label_score.to(device),
            added_score=self.added_score.to(device),
            all_masks=self.all_masks.to(device),
            all_pred=self.all_pred.to(device),
            baseline=self.baseline.to(device)
        )

    def cpu(self):
        return self.to(torch.device("cpu"))
    
    def shuffle(self):
        perm = torch.randperm(self.all_pred.shape[0]).to(self.all_pred.device)
        return MaskedRespData(
            baseline_score = self.baseline_score,
            label_score = self.label_score,
            added_score=self.added_score,
            all_masks=self.all_masks[perm],
            all_pred=self.all_pred[perm],
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


@functools.lru_cache(maxsize=None)
def get_tv_XTX(shape, rtv=True, ctv=True, norm=True):
    numel = shape[0]*shape[1]
    res = torch.zeros(numel, numel)
    count = 0
    if ctv:
        for idx in range(shape[0]):
            for jdx in range(shape[1]-1):
                count += 1
                idx_pos = idx*shape[0] + jdx 
                idx_neg = idx*shape[0] + jdx+1
                res[idx_pos, idx_pos] += 1.0
                res[idx_neg, idx_neg] += 1.0
                res[idx_pos, idx_neg] -= 1.0
                res[idx_neg, idx_pos] -= 1.0
    if rtv:
        for idx in range(shape[0]-1):
            for jdx in range(shape[1]):
                #count += 1
                idx_pos = idx*shape[0] + jdx
                idx_neg = (idx+1)*shape[0] + jdx
                res[idx_pos, idx_pos] += 1.0
                res[idx_neg, idx_neg] += 1.0
                res[idx_pos, idx_neg] -= 1.0
                res[idx_neg, idx_pos] -= 1.0

    if norm:
        res = res / torch.Tensor([count]).unsqueeze(0)
    return res

def optimize_ols(masks, responses, c_magnitude, c_tv, c_sample, c_weights=None):
    print("optimize_ols")
    masks = masks.cpu() * 1.0 
    assert 0 <= c_sample <= 1
    oshape = masks.shape[1:]

    if (c_sample < 1):
        masks = masks.unsqueeze(1)  
        masks_downsampled = F.interpolate(masks, scale_factor=0.5, mode='bilinear', align_corners=False)
        masks_downsampled = masks_downsampled.squeeze(1) 
        masks = masks_downsampled

    dshape = masks.shape[1:]
    Y = responses.cpu() / (oshape[0] * oshape[1])

    fmasks = masks.flatten(start_dim=1)

    
    weights = torch.sqrt(1/ (2 * fmasks.shape[0] * fmasks.sum(dim=1, keepdim=True)))
    if c_weights is not None:
        weights = weights * torch.sqrt(fmasks.shape[0] * c_weights / c_weights.sum()).unsqueeze(1)
    Xw = fmasks * weights
    Yw = Y * weights[:,0]
    XTXw = Xw.T @ Xw 
    XTY = Xw.T @ Yw

    ## reverting data generation numel factor
    tvXTX = get_tv_XTX(dshape)    
    XTX = XTXw + torch.eye(XTXw.shape[0]) *  c_magnitude / XTXw.shape[0]  + tvXTX*c_tv
    bb, _info = gmres(XTX.numpy(), XTY.numpy())
    msal = torch.Tensor(bb.reshape(*dshape)).unsqueeze(0)
    
    if oshape != dshape:
        msal = F.interpolate(msal.unsqueeze(0), size=oshape, mode='bilinear', align_corners=False)[0]
    return msal[0]



class CompExpCreator:

    def __init__(self, nmasks=500, segsize=64, batch_size=32, 
                 lr = 0.05, lr_step=0, lr_step_decay=0,
                 c_mask_completeness=1.0, c_completeness=0.1, 
                 c_smoothness=0, c_selfness=0.0, c_tv=1,
                 c_magnitude=0, c_positive=0, c_norm=False, c_activation=False,
                 c_compliment=False,
                 c_logistic = False,
                 c_logit = False,
                 avg_kernel_size=(5,5),
                 select_from=100, select_freq=10, select_del=0.5,
                 epochs=300, 
                 model_epochs=300, c_model=0,
                 c_sample = 0.5,
                 c_opt="Adam",
                 mgen=None,
                 desc = "MComp",                 
                 pprob = [0.5],
                 baseline_gen = ZeroBaseline(),                 
                 ext_desc = "",
                 cap_response = False,
                 force_desc = False,                 
                 **kwargs):
        
        assert type(segsize) == type(nmasks)
        if type(segsize) == int:
            segsize = [segsize]
            nmasks = [nmasks]
            pprob = [pprob]

        assert len(segsize) == len(nmasks)
        assert len(segsize) == len(pprob)

        self.segsize = segsize
        self.pprob = pprob
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
        self.c_positive = c_positive        
        self.c_compliment = c_compliment
        self.c_opt = c_opt
        self.c_sample = 0.5
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
        self.cap_response = cap_response        
        self.baseline_gen = baseline_gen
        if self.model_epochs == 0 or self.c_model == 0:
            self.model_epochs = 0
            self.c_model = 0
        self.ext_desc = ext_desc
        self.force_desc = force_desc
        self.c_logistic = c_logistic


    def description(self):
        if self.force_desc:
            return self.desc
        
        if self.epochs:
            opt_desc = f'{self.epochs}'
        else:
            opt_desc = 'OLS'

        if len(self.nmasks) == 1:
            desc = f"{self.desc}{self.ext_desc}_{self.nmasks[0]}_{self.segsize[0]}_{opt_desc}"
        else:
            desc = f"{self.desc}{self.ext_desc}_Mr_{opt_desc}"

        if self.epochs:
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

            if self.c_positive:
                desc += f"_p{self.c_positive}"

            if self.c_compliment:
                desc += f"_C"

            if self.c_model:
                desc += f"_mdl{self.c_model}"

            if self.c_selfness:
                desc += f"_sf{self.c_selfness}"
        else:
            if self.c_sample:
                desc += f"_s{self.c_sample}"

            if self.c_tv:
                desc += f"_tv{self.c_tv}"

            if self.c_magnitude:
                desc += f"_mgn{self.c_magnitude}"


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

        parts = list(zip(self.segsize, self.nmasks, self.pprob))        
        for segsize, nmasks, pprob  in parts:
            mgen = MaskedRespGen(segsize, mgen=self.mgen, baseline=baseline, ishape=me.shape, prob=pprob)            
            logging.debug(f"generating {nmasks} masks and responses")
            print(f"generating {nmasks} masks and responses segsize={segsize}")
            mgen.gen(fmdl, inp, nmasks, batch_size=self.batch_size)        
            all_masks_list += mgen.all_masks
            all_pred_list += mgen.all_pred
        logging.debug("Done generating masks")

        
        rfactor = inp.numel()        
        baseline_score = fmdl(baseline).detach().squeeze()
        label_score = fmdl(inp).detach().squeeze()

        device = me.device

        if self.c_logistic:
            norm = lambda x: x
            print("No normalization for logistic")
        elif logit:
            norm = lambda x: (torch.logit(x) + torch.log((1-baseline_score)/baseline_score)) * rfactor
        elif self.cap_response:
            norm = lambda x: torch.maximum((x - baseline_score), torch.zeros(1).to(device)) * rfactor
        else:
            print("## basic response")
            norm = lambda x: (x - baseline_score) * rfactor
                
        added_score = norm(label_score)
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
            all_pred_raw = (torch.concat(all_pred_list).to(device).squeeze()),
            baseline = baseline
        )

    def explain(self, me, inp, catidx, data=None, initial=None, callback=None):

        start_time = time.time() 
        if data is None:                        
            data = self.generate_data(me, inp, catidx, logit=self.c_logit)
            report_duration(start_time, me.arch, "SLOC_GEN")

        start_time_expl = time.time()

        if initial is None:
            #initial = torch.rand(inp.shape[-2:]).to(inp.device)
            #initial = (torch.randn(224,224)*0.2+1).abs()
            print("setting initial")
            if self.c_logistic:
                #assert False
                bs = torch.logit(data.label_score.cpu()) / (me.shape[0] * me.shape[1])
                initial = (torch.randn(me.shape[0],me.shape[1])*0.1+1) * bs                
                print("logistic initial", initial.mean())
            else:
                if self.c_compliment:
                    initial = (torch.randn(me.shape[0],me.shape[1])*0.1+1)
                    initial = initial *  data.added_score.cpu() / initial.sum()
                else:                
                    initial = (torch.randn(me.shape[0],me.shape[1])*0.1+1)
                    #initial = (torch.randn(me.shape[0],me.shape[1])*0.2+3)

        
        if self.epochs:
            fmdl = me.narrow_model(catidx, with_softmax=True)        
            
            sal = optimize_explanation(fmdl, inp, initial, data.all_masks, data.all_pred, score=data.added_score, 
                                    epochs=self.epochs, select_from=self.select_from, 
                                    select_freq=self.select_freq, select_del=self.select_del,
                                    model_epochs=self.model_epochs, lr=self.lr, c_opt=self.c_opt,
                                    avg_kernel_size=self.avg_kernel_size,
                                    c_completeness=self.c_completeness, c_smoothness=self.c_smoothness, 
                                    c_tv=self.c_tv, c_selfness=self.c_selfness,
                                    c_mask_completeness=self.c_mask_completeness,
                                    c_logistic=self.c_logistic,
                                    c_model=self.c_model,
                                    c_magnitude=self.c_magnitude,
                                    c_positive = self.c_positive,
                                    c_compliment = self.c_compliment,
                                    c_norm=self.c_norm, c_activation=self.c_activation,
                                    baseline=data.baseline, callback=callback)
        else:
            sal = optimize_ols(masks=data.all_masks, responses=data.all_pred, 
                               c_magnitude=self.c_magnitude, c_tv=self.c_tv, c_sample=self.c_sample)

        report_duration(start_time_expl, me.arch, "SLOC_OPT")
        
        
        return sal


class MultiCompExpCreator:

    def __init__(self, nmasks=500, mask_groups={"":{16:500,48:500}}, baselines=[ZeroBaseline()],
                 batch_size=32,
                 desc="MComp",
                 pprob = [None],
                 groups=[], 
                 acargs={}):
        self.mask_groups = mask_groups
        self.batch_size = batch_size
        self.baselines = baselines
        self.groups = groups
        self.desc = desc
        self.last_data = None
        self.pprob = pprob
        self.acargs = acargs
        logging.info("MultiCompExpCreator")

    def __call__(self, me, inp, catidx):        
        all_sals = {}
        logging.info(f"mask_groups: {len(self.mask_groups)}; group:{len(self.groups)}")
        for pprob in self.pprob:
            for bgen in self.baselines:
                
                seglimit = defaultdict(int)            
                for nm, maskspec in self.mask_groups.items():
                    for segsize, nmasks in maskspec.items():
                        seglimit[segsize] = max(seglimit[segsize], nmasks)
                
                seg_masks = {}
                for segsize, mlimit in seglimit.items():
                    if pprob is None:
                        tn = AutoCompExpCreator(nmasks=mlimit, segsize=segsize, batch_size=self.batch_size,
                                           baseline_gen=bgen, **self.acargs)
                        selected_pprob = tn.tune_pprob(segsize, me, inp, catidx)                        
                    else:
                        selected_pprob = pprob
                    
                    mgen = None
                    if segsize < 0:
                        mgen = SegMaskGen(inp, -segsize)
                    dc = CompExpCreator(nmasks=mlimit, segsize=segsize, batch_size=self.batch_size,
                                        baseline_gen=bgen, pprob=selected_pprob, mgen=mgen, **self.acargs)
                    
                    seg_masks[segsize] = dc.generate_data(me, inp, catidx)            
                
                for nm, maskspec in self.mask_groups.items():
                    logging.info(f"mask: {nm} {maskspec}")
                    data = MaskedRespData.join([seg_masks[segsize].subset(nmasks) for segsize, nmasks in maskspec.items()])
                                                            
                    # self.last_data = data
                    for kwargs in self.groups:
                        group_args = dict(nmasks=nmasks, segsize=segsize, batch_size=self.batch_size)
                        group_args.update(kwargs)                        
                        group_args['desc'] = self.desc + nm +  group_args.get('desc', '') + bgen.desc

                        algo = CompExpCreator(**group_args, ext_desc=f"{bgen.desc}{pprob}")
                        res = algo(me, inp, catidx, data=data)
                        all_sals.update(res)

        logging.info(f"generated: {list(all_sals.keys())}")
        return all_sals


class ProbSqMaskGen:

    def __init__(self, inner, prob):
        self.prob = prob
        self.inner = inner

    def gen_masks(self, nmasks):        
        probs = self.prob[0:nmasks]
        self.prob = self.prob[nmasks:]
        indexes = torch.arange(nmasks)
        
        all_masks = []
        all_indexes = []
        total_masks = 0
        idx = 0
        while total_masks < nmasks:
            #print("@@@", probs)
            #print("itr", idx)
            idx += 1
            remaining_masks = (nmasks - total_masks)
            masks = (self.inner.gen_masks_cont(remaining_masks) < probs.unsqueeze(1).unsqueeze(1) )
            is_valid = (masks.flatten(start_dim=1).sum(dim=1) > 0)
            num_valid = int(is_valid.sum())
            if num_valid == 0:
                continue
            valid_masks = masks[is_valid]
            all_masks.append(valid_masks)
            all_indexes.append(indexes[is_valid])
            indexes = indexes[~is_valid]
            probs = probs[~is_valid]
            total_masks += num_valid
            #print("done", total_masks)
        masks = torch.concat(all_masks)
        act_indexes = torch.concat(all_indexes)
        _, restore_indices = act_indexes.sort()
        return masks[restore_indices]
        



        return masks

class AutoCompExpCreator:

    def __init__(self, nmasks=[1000], segsize=[32], cap_response=False, tune_single_pass=True, 
                main_probs = [0.3, 0.4, 0.5, 0.6, 0.7],
                extra_probs = [0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.8],
                sampsize = 50,
                 **kwargs):
        self.nmasks = nmasks
        self.segsize = segsize        
        self.kwargs = kwargs
        self.cap_response = cap_response
        self.tune_single_pass = tune_single_pass
        self.main_probs =  main_probs
        self.extra_probs = extra_probs
        self.sampsize = sampsize
        
    
    def __call__(self, me, inp, catidx, callback=None, return_algo=False):
        start_time = time.time()
        pprob = [self.tune_pprob(segsize, me, inp, catidx) for segsize in self.segsize]
        logging.info(f"selected probs: ARCH,{me.arch},SEG,{','.join(map(str,self.segsize))},PROB,{','.join(map(str,pprob))}")
        report_duration(start_time, me.arch, "SLOC_TUNE")
        algo = CompExpCreator(nmasks=self.nmasks, segsize=self.segsize,         
                              cap_response=self.cap_response, pprob=pprob, **self.kwargs)
        if return_algo:
            return algo
        if callback:
            rv = algo.explain(me, inp, catidx, callback=callback)
        else:
            rv = algo(me, inp, catidx)
        report_duration(start_time, me.arch, "SLOC")
        return rv

    def get_prob_score(self, pprob, segsize, me, inp, catidx, sampsize=None):
        sampsize = sampsize or self.sampsize
        #logging.info(f"get_prob_score: {segsize}, {sampsize}, {pprob}")
        algo = CompExpCreator(desc="gen", segsize=segsize, nmasks=sampsize, cap_response=self.cap_response, pprob=pprob)    
        data = algo.generate_data(me, inp, catidx)         
        rv = float(data.all_pred.std().cpu())
        return rv

    def get_prob_score_list(self, probs, segsize, me, inp, catidx):
        pscore = lambda x: self.get_prob_score(x, segsize, me, inp, catidx)        
        return torch.tensor([pscore(x) for x in probs])

    def get_prob_score_list(self, pprob, segsize, me, inp, catidx, sampsize=50):
        #print("Checking", segsize, pprob)
        prob_list = []
        for x in pprob:
            prob_list += ([x] * sampsize)

        if segsize > 0: 
            inner = SqMaskGen(segsize=segsize, mshape=me.shape)
        else:
            inner = SegMaskGen(inp, n_segments=-segsize)
        mgen = ProbSqMaskGen(inner, prob=torch.Tensor(prob_list))
        
        algo = CompExpCreator(desc="gen", segsize=[segsize], nmasks=[sampsize*len(pprob)], mgen=mgen,
                              pprob=[torch.Tensor(prob_list)], batch_size=len(prob_list))    
        data = algo.generate_data(me, inp, catidx)  

        rv = []
        for idx, prob in enumerate(pprob):
            score = float(data.all_pred[(idx*sampsize):((idx+1)*sampsize)].std().cpu())
            rv.append(score)
        return torch.Tensor(rv)

    def tune_pprob(self, segsize, me, inp, catidx, single_pass=True):
        logging.info(f"tune_pprob: {segsize}")        
        pscore = lambda x: self.get_prob_score(x, segsize, me, inp, catidx)
        main_probs = self.main_probs
        extra_probs = self.extra_probs

        if single_pass: 
            main_probs = main_probs + extra_probs

        main_scores = self.get_prob_score_list(main_probs, segsize, me, inp, catidx)
        
        
        if single_pass:
            all_probs = torch.Tensor(main_probs)
            all_scores = main_scores
        else:
            foc = main_probs[int(main_scores.argmax())]
            aux_probs = [x for x in extra_probs if (foc - 0.15 <= x <= foc + 0.15)]    
            aux_scores = self.get_prob_score_list(aux_probs, segsize, me, inp, catidx)
            all_probs = torch.Tensor(main_probs + aux_probs)
            all_scores = torch.concat([main_scores, aux_scores])

        rv = float(all_probs[int(all_scores.argmax())])
        print(f"tuned: {segsize} - > {rv}")
        return rv

class MulCompExpCreator(AutoCompExpCreator):

    def __init__(self, mode=["mean"], seq=False, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.seq = seq

    def flatten(self, items):
        rv = []
        for x in items:
            if type(x) in [list, set]:
                rv += self.flatten(x)
            else:
                rv.append(x)
        return rv

    def __call__(self, me, inp, catidx):
        all_segsize = list(set(self.flatten(self.segsize)))
        pprob_dict = { segsize : self.tune_pprob(segsize, me, inp, catidx, single_pass=self.tune_single_pass) 
                      for segsize in all_segsize }
        #pprob = [self.tune_pprob(segsize, me, inp, catidx) for segsize in self.segsize]
        #logging.info(f"selected probs: ARCH,{me.arch},SEG,{','.join(map(str,self.segsize))},PROB,{','.join(map(str,pprob))}")


        res = {}
        exp_list = []
        desc = None
        for idx, segsize in enumerate(self.segsize):

            pprob = [pprob_dict[x] for x in segsize]
            c_positive = False # (self.mode == "mul")
            algo = CompExpCreator(
                nmasks=self.nmasks, segsize=segsize, 
                cap_response=self.cap_response, pprob=pprob, c_positive=c_positive, **self.kwargs)        
            desc = (desc or algo.description())

            cexp = algo.explain(me,inp, catidx).cpu().unsqueeze(0)
            if self.mode == "prod":
                cexp = torch.maximum(cexp, torch.zeros(1))
            exp_list.append(cexp)
            
            for mode in self.mode:
                if mode in ["mean", "median","min"]:
                    stacked = torch.stack(exp_list, dim=0)
                    if mode == "median":
                        exp, _ = torch.median(stacked, dim=0) 
                    elif mode =="mean":
                        exp = torch.mean(stacked, dim=0)
                    elif mode == "min":
                        exp, _ = torch.min(stacked, dim=0)
                elif mode == "prod":
                    exp = exp_list[0]
                    for cexp in exp_list[1:]:
                        exp = exp * cexp * (exp > 0) * (cexp > 0)
                res[f"{desc}_{mode}_{idx+1}"] = exp

        return res
        

class MProbCompExpCreator:

    def __init__(self, nmasks=[1000], segsize=[32], **kwargs):
        self.kwargs = kwargs        
        self.nmasks = nmasks
        self.segsize = segsize

    def __call__(self, me, inp, catidx):
        if 'vit_small' in me.arch:
            pprob = [0.3] * len(self.segsize)
        elif 'vit_base' in me.arch:
            pprob = [0.2] * len(self.segsize)
        elif me.arch == 'resnet50':
            pprob = [0.6] * len(self.segsize)
        elif me.arch == 'densenet201':
            pprob = [0.6] * len(self.segsize)
        else:
            assert False, f"Unexpected arch {me.arch}"
        start_time = time.time()
        algo = CompExpCreator(nmasks=self.nmasks, segsize=self.segsize, pprob=pprob, **self.kwargs)
        rv = algo(me, inp, catidx)
        report_duration(start_time, me.arch, "SLOCxP")
        return rv



class ProbRangeCompExpCreator:

    def __init__(self, nmasks=[500,500], segsize=[32,56], min_prob=0.2, max_prob=0.8, **kwargs):
        self.nmasks = nmasks
        self.segsize = segsize        
        self.min_prob = min_prob
        self.max_prob = max_prob

        self.kwargs = kwargs

    def __call__(self, me, inp, catidx):

        all_data = []
        for nmasks, segsize in zip(self.nmasks, self.segsize):            
            pprob = self.min_prob + (self.max_prob - self.min_prob) * (torch.arange(nmasks) / (nmasks-1.0))
            mgen = ProbSqMaskGen(segsize=segsize, mshape=me.shape, prob=pprob)
            algo = CompExpCreator(
                desc="gen", segsize=[segsize], nmasks=[nmasks],
                pprob=[pprob], mgen=mgen)    
            data = algo.generate_data(me, inp, catidx)  
            all_data.append(data)

        all_data = MaskedRespData.join(all_data)            
        algo = CompExpCreator(nmasks=self.nmasks, segsize=self.segsize, pprob=[0.5]*len(self.nmasks), **self.kwargs)        
        return algo(me,inp,catidx, data=all_data)
        

class SegSlocExpCreator:
    def __init__(self, desc, seg_list=[], sq_list=[], **kwargs):
        self.seg_list = seg_list
        self.sq_list = sq_list
        self.desc = desc
        self.kwargs = kwargs

    def __call__(self, me, inp, catidx):
        all_data = []
        for segsize, nmasks, pprob in self.sq_list:
            algo = CompExpCreator(
                desc=self.desc, segsize=[segsize], nmasks=[nmasks], pprob=[pprob], force_desc=True, **self.kwargs)
            idata = algo.generate_data(me, inp, catidx)
            all_data.append(idata)
            
        for nsegs, nmasks, pprob in self.seg_list:
            sg = SegMaskGen(inp, nsegs)
            algo = CompExpCreator(
                desc=self.desc, segsize=[0], nmasks=[nmasks], pprob=[pprob], mgen=sg, force_desc=True, **self.kwargs)
            idata = algo.generate_data(me, inp, catidx)
            all_data.append(idata)

        data = MaskedRespData.join(all_data)
        sal = algo(me, inp, catidx, data=data)
        return sal
    

            
        
