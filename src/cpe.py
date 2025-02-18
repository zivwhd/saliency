import numpy as np
from skimage.segmentation import slic,mark_boundaries
import torch
import cv2, logging
from scipy.spatial import Voronoi
import torch.nn.functional as F
import torch.nn as nn

#try:
#    from tqdm import tqdm
#except:
#    tqdm = lambda x: x
tqdm = lambda x: x
#from tqdm import tqdm

import pdb




def gen_seg_masks_cont(segments, nmasks, width = 224, height = 224):

    masks = np.zeros((nmasks, height, width), dtype=np.float32)
    for idx in range(nmasks):#(32 masks)                            
        w_crop = np.random.randint(0, segments.shape[1] - width)
        h_crop = np.random.randint(0, segments.shape[1] - height)
    
        wseg = segments[h_crop:height + h_crop, w_crop:width + w_crop]
        items = np.unique(wseg)
        nitems = items.shape[0]
        
        #prob = 0.5
        #selection = (np.random.random(nitems) > prob)        
        for sid in items:
            masks[idx][wseg == sid] = np.random.random()
    return masks

def gen_seg_masks(segments, nmasks, prob=0.5, width = 224, height = 224):
    masks = gen_seg_masks_cont(segments, nmasks, width=width, height=height)
    #return masks
    return (masks < prob) 

def masked_output(model, inp, masks):    
    masks = masks.to(inp.device)    
    masked_input = inp * masks.unsqueeze(1)
    out = model(masked_input) ## CHNG
    return out

def mask_step(model, inp, gen_masks, itr, batch_size=32):
    all_out = []
    all_masks = []
    for idx in range(itr):        
        masks = gen_masks(batch_size)
        if type(masks) == np.ndarray:
            masks = torch.tensor(masks)
        out = masked_output(model, inp, masks)
        all_out.append(out.cpu())
        all_masks.append(masks.cpu())
    
    return torch.concatenate(all_out), torch.concatenate(all_masks)

class SqMaskGen:

    def __init__(self, segsize, mshape, efactor=4, prob=0.5):
        base = np.zeros((mshape[0] * efactor, mshape[1] * efactor ,3), np.int32)
        n_segments = base.shape[0] * base.shape[1] / (segsize * segsize)
        self.setsize = n_segments
        self.segments = torch.tensor(slic(base,n_segments=n_segments,compactness=1000,sigma=1), dtype=torch.int32)
        self.nelm = torch.unique(self.segments).numel()
        self.prob = prob
        self.mshape = mshape
    
    def gen_masks_(self, nmasks):
        return gen_seg_masks(self.segments, nmasks, width=self.mshape[1], height=self.mshape[0])
    
    def gen_masks_cont_(self, nmasks):
        return gen_seg_masks_cont(self.segments, nmasks, width=self.mshape[1], height=self.mshape[0])

    def gen_masks(self, nmasks):        
        masks = (self.gen_masks_cont(nmasks) < self.prob)        
        return masks
    
    def gen_masks_cont(self, nmasks):
        
        height, width= self.mshape        
        #masks = torch.zeros((nmasks, height, width))

        step = self.nelm
        nelm = step * nmasks
        rnd = torch.rand(2+nelm)
        stt = []
        for idx in range(nmasks):
            w_crop = torch.randint(self.segments.shape[1]- width, (nmasks,))
            h_crop = torch.randint(self.segments.shape[0]- height, (nmasks,))
            wseg = self.segments[h_crop[idx]:height + h_crop[idx], w_crop[idx]:width + w_crop[idx]] + step * idx
            stt.append(wseg)

        parts = torch.stack(stt)
        
        return rnd[parts.view(-1)].view(parts.shape)



    
class IpwGenBase:
    def __init__(self):
        self.saliency = None
        self.weights = None

    def get_ips_sal(self, clip=0.1):
        num_cat = self.saliency.shape[0]
        assert (num_cat in (32, 128))
        idx = torch.arange(num_cat)
        treatment_sal = self.saliency[(idx & 1) == 1]
        control_sal = self.saliency[(idx & 1) == 0]
        p_treatment_weights = self.weights[(idx & 1) == 1]
        p_control_weights = self.weights[(idx & 1) == 0]

        if clip is not None:
            t_clipping = torch.tensor([clip])    
            c_clipping = torch.tensor([clip])    
        else:
            clipping = torch.tensor([0.1]) 
            t_clipping = torch.max((1 / (p_treatment_weights + 0.1)**0.5), clipping)
            c_clipping = torch.max((1 / (p_control_weights + 0.1)**0.5), clipping)

        treatment_prob =  torch.max(p_treatment_weights / (p_treatment_weights + p_control_weights + 1), t_clipping)
        control_prob =  torch.max(p_control_weights / (p_treatment_weights + p_control_weights + 1), c_clipping)

        ipw = (
            ((treatment_sal / treatment_prob).sum(dim=0) / (p_treatment_weights / treatment_prob).sum(dim=0)) -
            ((control_sal / control_prob).sum(dim=0) / (p_control_weights / control_prob).sum(dim=0)) ).unsqueeze(0)            
        
        return ipw[0]

    def get_ate_sal(self):
        num_cat = self.saliency.shape[0]
        idx = torch.arange(num_cat)
        treatment = self.saliency[(idx & 1)==1].sum(dim=0) / self.weights[(idx & 1)==1].sum(dim=0)
        ctrl = self.saliency[(idx & 1)==0].sum(dim=0) / self.weights[(idx & 1)==0].sum(dim=0)
        ate = treatment-ctrl
        return ate

    def get_exp_sal(self):
        num_cat = self.saliency.shape[0]
        idx = torch.arange(num_cat)
        treatment = self.saliency[(idx & 1)==1].sum(dim=0) / self.weights[(idx & 1)==1].sum(dim=0)
        return treatment        

class IpwGen(IpwGenBase):

    def __init__(self, segsize=68, ishape = (224,224),
                 MaskGen = SqMaskGen, degrees = [0, 90, 180, 270]):
        super().__init__()
        self.segsize = segsize
        self.pad = self.segsize // 2
        self.ishape = ishape
        self.mgen = MaskGen(segsize, mshape=(ishape[0] + self.pad * 2, ishape[1] + self.pad * 2))
        self.num_cat = 2 * (2**len(degrees))
        self.degrees = degrees

    def gen(self, model, inp, nmasks, batch_size=32, **kwargs):
        with torch.no_grad():
            self.gen_(model=model, inp=inp, itr=nmasks//batch_size, batch_size=batch_size, **kwargs)
            if nmasks % batch_size:
                self.gen_(model=model, inp=inp, itr=1, batch_size=nmasks % batch_size, **kwargs)
        
    def gen_(self, model, inp, itr=125, batch_size=32):        
        h = self.ishape[0]
        w = self.ishape[1]
        pad = self.pad
        deg = torch.tensor(self.degrees)
        xoffs = (pad * torch.sin(2*torch.pi * deg /360)).int().tolist()
        yoffs = (pad * torch.cos(2*torch.pi * deg /360)).int().tolist()

        for idx in tqdm(range(itr)):
            exp_masks = self.gen_masks(batch_size)
            if type(exp_masks) == np.ndarray:
                exp_masks = torch.tensor(exp_masks)
            exp_masks = exp_masks.unsqueeze(1)
            
            mmasks = exp_masks[..., self.pad:self.pad+h, self.pad:self.pad+w]
            out = masked_output(model, inp, mmasks.squeeze(1)).cpu()            
            
            mout = out.unsqueeze(-1).unsqueeze(-1)

            #cat = ((mmasks > 0.5) + 
            #    (exp_masks[:,:,2*pad:2*pad+w, pad:pad+h] > 0.5) * 2 +
            #    (exp_masks[:,:,0:w, pad:pad+h] > 0.5) * 4 +
            #    (exp_masks[:,:,pad:pad+w, 2*pad:2*pad+h] > 0.5) * 8 +
            #    (exp_masks[:,:,pad:pad+w, 0:h] > 0.5) * 16)

            cat = (mmasks > 0.5) * 1
            for idx in range(len(xoffs)):
                cat += (exp_masks[:,:,pad+yoffs[idx]:pad+h+yoffs[idx], pad+xoffs[idx]:pad+w+xoffs[idx]] > 0.5) * 2 * 2**idx
            
            mbin = (
                torch.arange(self.num_cat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) ==
                cat.unsqueeze(0))
            
            saliency = (mout.unsqueeze(0) * mbin).sum(dim=1)
            weights = mbin.sum(dim=1)
            if self.saliency is None:
                self.saliency = saliency
                self.weights = weights
            else:
                self.saliency += saliency
                self.weights += weights



    def gen_masks(self, batch_size):
        return self.mgen.gen_masks(batch_size)



class IpwSalCreator:

    multi_target = True
    def __init__(self, desc, nmasks, batch_size=32, clip=[0.1], ipwg = IpwGen, with_softmax=False, **kwargs):
        self.nmasks = nmasks
        self.desc = desc
        self.clip = clip
        self.batch_size = batch_size
        self.ipwg = ipwg
        self.with_softmax = with_softmax
        self.kwargs = kwargs
        

    def __call__(self, me, inp, catidx):
        ipwg = self.ipwg(**self.kwargs)
        total = 0
        res = {}
        for nmasks in self.nmasks:
            added_nmasks = nmasks - total
            total = nmasks
            logging.debug(f"IpSalCreator: nmasks={nmasks}; added = {added_nmasks}")
            ipwg.gen(me.narrow_model(catidx, with_softmax=self.with_softmax), inp, nmasks=added_nmasks, batch_size=self.batch_size)            
            res[f"{self.desc}_{nmasks}_ate"] = ipwg.get_ate_sal()
            for clip in self.clip:
                res[f"{self.desc}_{nmasks}_ipw_{clip}"] = ipwg.get_ips_sal(clip)
        logging.debug(f"IpSalCreator: total masks={total};")
        return res 
    

class SelectKthLogitSoftmax(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss  = nn.CrossEntropyLoss()
        assert type(self.k) == int

    def forward(self, x):                
        values = torch.stack([x, self.softmax(x)], dim=-1)
        result = values[:,self.k,:]            
        return result

class AblIpwSalCreator:

    def __init__(self, desc, nmasks, batch_size=32, clip=[0.1], ipwg = IpwGen, **kwargs):
        self.nmasks = nmasks
        self.desc = desc
        self.clip = clip
        self.batch_size = batch_size
        self.ipwg = ipwg        
        self.kwargs = kwargs
        

    def __call__(self, me, inp, catidx):
        ipwg = self.ipwg(**self.kwargs)
        total = 0
        res = {}
        for nmasks in self.nmasks:
            added_nmasks = nmasks - total
            total = nmasks
            logging.debug(f"IpSalCreator: nmasks={nmasks}; added = {added_nmasks}")
            model = nn.Sequential(me.model, SelectKthLogitSoftmax(catidx))
            ipwg.gen(model, inp, nmasks=added_nmasks, batch_size=self.batch_size)

            ate_sal = ipwg.get_ate_sal()            
            res[f"{self.desc}_{nmasks}_ate_logit"] = ate_sal[0:1]
            res[f"{self.desc}_{nmasks}_ate_prob"] = ate_sal[1:2]
            exp_sal = ipwg.get_exp_sal()
            res[f"{self.desc}_{nmasks}_exp_logit"] = exp_sal[0:1]
            res[f"{self.desc}_{nmasks}_exp_prob"] = exp_sal[1:2]
            for clip in self.clip:
                ips_sal = ipwg.get_ips_sal(clip)
                res[f"{self.desc}_{nmasks}_ipw_{clip}_logit"] = ips_sal[0:1]
                res[f"{self.desc}_{nmasks}_ipw_{clip}_prob"] = ips_sal[1:2]

        logging.debug(f"IpSalCreator: total masks={total};")
        return res 



###
class SimpGen:
    def __init__(self, segsize=68, ishape = (224,224), force_mask=None, collect_masks=False):
        self.treatment = None
        self.ctrl = None
        self.treatment2 = None
        self.ctrl2 = None
        self.total_masks = 0        
        self.weights = None

        self.segsize = segsize
        self.pad = self.segsize // 2
        self.ishape = ishape
        self.mgen = SqMaskGen(segsize, mshape=ishape)
        self.weights = torch.zeros(ishape)
        self.sals = torch.zeros(ishape)
        self.sals2 = torch.zeros(ishape)
        self.force_mask = force_mask

        self.collect_masks = collect_masks
        self.all_masks = []
        self.all_pred = []

    def gen_(self, model, inp, itr=125, batch_size=32):
        
        h = self.ishape[0]
        w = self.ishape[1]
        pad = self.pad

        force_mask = self.force_mask
        if force_mask is not None:
            force_mask = force_mask.unsqueeze(0).to(inp.device)

        for idx in tqdm(range(itr)):
            masks = self.mgen.gen_masks(batch_size)
            self.total_masks += masks.shape[0]
            #if type(exp_masks) == np.ndarray:
            #    exp_masks = torch.tensor(exp_masks)
            #masks = masks.unsqueeze(1)            
            dmasks = masks.to(inp.device)    
            if force_mask is not None:
                dmasks = dmasks | force_mask

            out = masked_output(model, inp, dmasks)
            mout = out.unsqueeze(-1).unsqueeze(-1)

            if self.collect_masks:
                self.all_masks.append(masks.cpu())
                self.all_pred.append(mout.cpu())
            streatment = mout * dmasks.unsqueeze(1)
            sctrl = mout * (1 - 1.0 * dmasks.unsqueeze(1))

            treatment = streatment.sum(dim=0)
            ctrl = sctrl.sum(dim=0)

            treatment2 = (streatment*streatment).sum(dim=0)
            ctrl2 = (sctrl*sctrl).sum(dim=0)
              
            weights = dmasks.sum(dim=0, keepdim=True)

            if self.treatment is None:
                self.treatment = treatment
                self.ctrl = ctrl
                self.treatment2 = treatment2
                self.ctrl2 = ctrl2
                self.weights = weights
            else:
                self.treatment += treatment
                self.ctrl += ctrl
                self.treatment2 += treatment2
                self.ctrl2 += ctrl2
                self.weights += weights
            

    def gen(self, model, inp, nmasks, batch_size=32, **kwargs):        
        with torch.no_grad():
            self.gen_(model=model, inp=inp, itr=nmasks//batch_size, batch_size=batch_size, **kwargs)
            if nmasks % batch_size:
                self.gen_(model=model, inp=inp, itr=1, batch_size=nmasks % batch_size, **kwargs)

    def var(self, values, values2, weights):
        return (values2 / weights) - (values * values) / (weights * weights)
    
    def get_ate_sal_i(self):
        ctrl_weights = (self.total_masks - self.weights)
        ate = (self.treatment /  self.weights) - (self.ctrl / ctrl_weights)
        
        treatment_var = self.var(self.treatment, self.treatment2, self.weights)
        ctrl_var = self.var(self.ctrl, self.ctrl2, ctrl_weights)
        ate_var = treatment_var + ctrl_var

        return ate, ate_var
    
    def get_ate_sal(self):
        return self.get_ate_sal_i()[0]

def gsobel(K):    
    assert K % 2 == 1
    arng = torch.arange(K, dtype=torch.float32)
    offs = arng-arng.mean()
    dist = offs.abs()    
    return torch.nan_to_num(offs / (dist.unsqueeze(0)**2 +  dist.unsqueeze(1)**2),0)

sblx = gsobel(31)
sbly = sblx.transpose(0,1)

class RelIpwGen(SimpGen):
    def __init__(self, segsize=64, ishape = (224,224)):
        super().__init__(segsize=segsize, ishape=ishape, force_mask=None, collect_masks=True)
        #def __init__(self, segsize=68, ishape = (224,224), force_mask=None, collect_masks=False):

        logging.debug(f"RelIpwGen {segsize}")
        sobelK = ((segsize // 2) - 1 + (segsize // 2) % 2)
        self.rdist = sobelK
        self.sblx = gsobel(sobelK)
        self.sbly = sblx.transpose(0,1)

    def get_ips_sal(self, clip=0.1):
        
        ## first get simple ate sal - to find potential confounder
        ate = self.get_ate_sal()
        isal = ate.cpu()

        ## find saliency gradient - the potnetial confounder is upward
        ctx = F.conv2d(isal.unsqueeze(0), sblx.unsqueeze(0).unsqueeze(0), padding="same")[0,0]
        cty = F.conv2d(isal.unsqueeze(0), sbly.unsqueeze(0).unsqueeze(0), padding="same")[0,0]
        csz = (ctx**2 + cty**2).sqrt()
        cszbar = torch.quantile(csz, 0.01)

        ## calculate offsets 
        offsx = (ctx*self.rdist/csz).to(torch.int32)
        offsy = (cty*self.rdist/csz).to(torch.int32)

        # offset to indexes, validation
        H, W = self.ishape
        idxx = offsx + torch.arange(W).unsqueeze(0)
        idxy = offsy + torch.arange(H).unsqueeze(1) ## PUSH_ASSERT
        isok = ((csz > cszbar) & (idxx >= 0) & (idxx < W) & (idxy >= 0) & (idxy < H))
        
        ## flatten indexes
        confidx = torch.maximum(torch.minimum(idxx.flatten()+idxy.flatten()*H, torch.tensor(H*W-1)), torch.tensor([0]))

        icats = torch.arange(4).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gidx = None
        prev_bmask_shape = None
        s_saliency = None
        s_weights = None

        # iterate over masks, and acummulate alienct, weights in bins
        logging.debug(f"number of batches: {len(self.all_masks)}")
        for idx, bmasks in enumerate(self.all_masks):
    
            if bmasks.shape != prev_bmask_shape:
                gidx = ((torch.arange(bmasks.shape[0]) * confidx.numel()).unsqueeze(1) + confidx.unsqueeze(0)).flatten()
                prev_bmask_shape = bmasks.shape
                
            ref = bmasks.flatten().gather(0, gidx).reshape(bmasks.shape)
            mout = self.all_pred[idx]
            #print(mout)
            #print(bmasks.shape)
            ref = bmasks.flatten().gather(0, gidx).reshape(bmasks.shape)
            cat = ((bmasks > 0.5) * 1 + (ref > 0.5) * 2).unsqueeze(0)
            
            mbin = (icats == cat)
            #print(cat.shape, mout.shape, mbin.shape)
            saliency = (mout.squeeze(1).unsqueeze(0) * mbin).sum(dim=1)
            #print(mbin.shape)
            weights = mbin.sum(dim=1)    
            if s_saliency is None:
                s_saliency = saliency
                s_weights = weights
            else:
                s_saliency += saliency
                s_weights += weights

        cidx = torch.arange(4)
        treatment_sal = s_saliency[(cidx & 1) == 1]  ## dim:0 is of size 2
        control_sal = s_saliency[(cidx & 1) == 0]

        p_treatment_weights = s_weights[(cidx & 1) == 1]
        p_control_weights = s_weights[(cidx & 1) == 0]
        
        t_clipping = torch.tensor([clip])    
        c_clipping = torch.tensor([clip])    

        treatment_prob =  torch.max(p_treatment_weights / (p_treatment_weights + p_control_weights + 1), t_clipping)
        control_prob =  torch.max(p_control_weights / (p_treatment_weights + p_control_weights + 1), c_clipping)

        ipw = (
            ((treatment_sal / treatment_prob).sum(dim=0) / (p_treatment_weights / treatment_prob).sum(dim=0)) -
            ((control_sal / control_prob).sum(dim=0) / (p_control_weights / control_prob).sum(dim=0)) ).unsqueeze(0)            

        return ipw




        