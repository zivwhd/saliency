import numpy as np
from skimage.segmentation import slic,mark_boundaries
import torch
import cv2, logging
from scipy.spatial import Voronoi

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

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

    def __init__(self, segsize, mshape, efactor=4):
        base = np.zeros((mshape[0] * efactor, mshape[1] * efactor ,3), np.int32)
        n_segments = base.shape[0] * base.shape[1] / (segsize * segsize)
        self.setsize = n_segments
        self.segments = torch.tensor(slic(base,n_segments=n_segments,compactness=1000,sigma=1), dtype=torch.int32)
        self.nelm = torch.unique(self.segments).numel()
        self.mshape = mshape
    
    def gen_masks_(self, nmasks):
        return gen_seg_masks(self.segments, nmasks, width=self.mshape[1], height=self.mshape[0])
    
    def gen_masks_cont_(self, nmasks):
        return gen_seg_masks_cont(self.segments, nmasks, width=self.mshape[1], height=self.mshape[0])

    def gen_masks(self, nmasks):
        return (self.gen_masks_cont(nmasks) < 0.5)
    
    def gen_masks_cont(self, nmasks):
        
        height, width= self.mshape        
        #masks = torch.zeros((nmasks, height, width))

        rnd = torch.rand(1+self.nelm)
        stt = []
        for idx in range(nmasks):
            w_crop = torch.randint(self.segments.shape[1]- width, (nmasks,))
            h_crop = torch.randint(self.segments.shape[0]- height, (nmasks,))
            wseg = self.segments[h_crop[idx]:height + h_crop[idx], w_crop[idx]:width + w_crop[idx]]
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
        logging.debug(f"IpwGen.gen_ itr={itr} itr={batch_size}")
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
        logging.debug(f"IpwGen.gen_ weights={(self.weights * 1.0).mean()}")


    def gen_masks(self, batch_size):
        return self.mgen.gen_masks(batch_size)



class IpwSalCreator:

    def __init__(self, desc, nmasks, batch_size=32, clip=[0.1], ipwg = IpwGen, **kwargs):        
        self.nmasks = nmasks
        self.desc = desc
        self.clip = clip
        self.batch_size = batch_size
        self.ipwg = IpwGen
        self.kwargs = kwargs
        

    def __call__(self, me, inp, catidx):
        ipwg = self.ipwg(**self.kwargs)
        total = 0
        res = {}
        for nmasks in self.nmasks:
            added_nmasks = nmasks - total
            total = nmasks
            logging.debug(f"IpSalCreator: nmasks={nmasks}; added = {added_nmasks}")
            ipwg.gen(me.narrow_model(catidx), inp, nmasks=added_nmasks, batch_size=self.batch_size)            
            res[f"{self.desc}_{nmasks}_ate"] = ipwg.get_ate_sal()
            for clip in self.clip:
                res[f"{self.desc}_{nmasks}_ipw_{clip}"] = ipwg.get_ips_sal(clip)
        logging.debug(f"IpSalCreator: total masks={total};")
        return res 