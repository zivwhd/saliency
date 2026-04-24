
import torch
import numpy as np
import functools
from skimage.segmentation import slic,mark_boundaries
import torch.nn.functional as F
from scipy.sparse.linalg import cg, gmres, lsqr

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

def optimize_ols(masks, responses, c_magnitude, c_tv, c_sample, c_weights=None, c_with_bias=False):
    print(f"optimize_ols: bias={c_with_bias}")
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

    if c_with_bias:
        bias_col = torch.ones(fmasks.size(0), 1, device=fmasks.device, dtype=fmasks.dtype)
        fmasks = torch.cat([fmasks, bias_col], dim=1)

    weights = torch.sqrt(1/ (2 * fmasks.shape[0] * fmasks.sum(dim=1, keepdim=True)))
    if c_weights is not None:
        weights = weights * torch.sqrt(fmasks.shape[0] * c_weights / c_weights.sum()).unsqueeze(1)
    Xw = fmasks * weights
    Yw = Y * weights[:,0]
    XTXw = Xw.T @ Xw 
    XTY = Xw.T @ Yw

    ## reverting data generation numel factor
    tvXTX = get_tv_XTX(dshape)    
    if c_with_bias:
        tmp = torch.zeros(XTXw.shape)
        tmp[1:,1:] = tvXTX
        tvXTX = tmp

    XTX = XTXw + torch.eye(XTXw.shape[0]) *  c_magnitude / XTXw.shape[0]  + tvXTX*c_tv
    bb, _info = gmres(XTX.numpy(), XTY.numpy())
    if c_with_bias:
        bb = bb[:-1]
    msal = torch.Tensor(bb.reshape(*dshape)).unsqueeze(0)
    
    if oshape != dshape:
        msal = F.interpolate(msal.unsqueeze(0), size=oshape, mode='bilinear', align_corners=False)[0]
    return msal[0]




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


class BaseSampExpCreator:

    def __init__(self, desc, c_tv=100, c_magnitude=50, c_with_bias=False, 
                 c_sample=0.5):
        self.desc = desc
        self.c_tv = c_tv
        self.c_magnitude = c_magnitude
        self.c_with_bias = c_with_bias
        self.c_sample = c_sample
        
    def description(self):
        return f"{self.desc}.{self.c_tv}.{self.c_magnitude}"
    
    def explain(self, me, inp, catidx, data=None):
        
        if data is None:                        
            data = self.generate_data(me, inp, catidx)

        sal = optimize_ols(
            masks=data.all_masks, responses=data.all_pred, 
            c_magnitude=self.c_magnitude, c_tv=self.c_tv, c_sample=self.c_sample,
            c_with_bias=self.c_with_bias)
        
        return sal.cpu()


    def generate_data(self, me, inp, catidx):
        raise NotImplementedError()
    

    def __call__(self, me, inp, catidx, data=None):
        desc = self.description()
        sal = self.explain(me, inp, catidx, data=data)
        csal = sal.cpu().unsqueeze(0)

        return {desc : csal}


#### Square patches


class SquareMaskGen:

    def __init__(self, segsize, mshape, efactor=4, prob=0.5, fcrop=None):
        base = np.zeros((mshape[0] * efactor, mshape[1] * efactor ,3), np.int32)
        n_segments = base.shape[0] * base.shape[1] / (segsize * segsize)
        self.setsize = n_segments
        self.segments = torch.tensor(slic(base,n_segments=n_segments,compactness=1000,sigma=1), dtype=torch.int32)
        self.nelm = torch.unique(self.segments).numel()
        self.prob = prob
        self.mshape = mshape
        self.fcrop = fcrop
    
    def gen_masks(self, nmasks):
        return (self.gen_masks_cont(nmasks) < self.prob)
        
    def gen_masks_cont(self, nmasks):
        
        height, width= self.mshape        
        #masks = torch.zeros((nmasks, height, width))

        step = self.nelm
        nelm = step * nmasks
        rnd = torch.rand(2+nelm)
        stt = []
        if self.fcrop is not None:
            w_crop = self.fcrop[1] * torch.ones(nmasks, dtype=torch.int32)
            h_crop = self.fcrop[0] * torch.ones(nmasks, dtype=torch.int32)
            
        for idx in range(nmasks):            
            if self.fcrop is None:                                
                w_crop = torch.randint(self.segments.shape[1]- width, (nmasks,))
                h_crop = torch.randint(self.segments.shape[0]- height, (nmasks,))            
            wseg = self.segments[h_crop[idx]:height + h_crop[idx], w_crop[idx]:width + w_crop[idx]] + step * idx
            stt.append(wseg)

        parts = torch.stack(stt)
        
        return rnd[parts.view(-1)].view(parts.shape)

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

    @staticmethod
    def generate(model, inp, mgen, nmasks, batch_size=32, **kwargs):
        mrgen = MaskedRespGen(mgen=mgen, **kwargs)
        mrgen.gen(model, inp, nmasks, batch_size=batch_size)
        return mrgen.all_masks, mrgen.all_pred

    def __init__(self, ishape = (224,224),
                 mgen=None, baseline=None, prob=0.5):

        self.ishape = ishape
        self.mgen = mgen

        if baseline is None:
            self.baseline = torch.zeros(ishape)
        else:
            self.baseline = baseline

        self.all_masks = []
        self.all_pred = []
        self.num_masks = 0

    @staticmethod
    @torch.no_grad()
    def generate_masks_robust(mgen, nmasks):
        num_masks = 0
        all_masks = []
        while num_masks < nmasks:                
            remaining = nmasks - num_masks            
            masks = mgen.gen_masks(remaining)
            is_valid = (
                (masks.flatten(start_dim=1).sum(dim=1) > 0) &
                ((masks.flatten(start_dim=1)*1.0).mean(dim=1) < 1))

            #print("###", is_valid.sum(), remaining)
            if (not any(is_valid)):
                continue
            masks = masks[ is_valid ]
            all_masks.append(masks)
            num_masks += is_valid.sum()
        return all_masks        


    @staticmethod
    @torch.no_grad()    
    def generate_masks_multi(mgen_list, nmasks_list):
        all_masks = []
        for mgen, nmasks in zip(mgen_list, nmasks_list):
            all_masks += MaskedRespGen.generate_masks_robust(mgen, nmasks)
        return torch.concat(all_masks)

    @staticmethod
    @torch.no_grad()
    def generate_predictions(model, inp, baseline, masks, batch_size=32):
                
        baseline = baseline.to(inp.device)        
        all_pred = []
        for idx in range(0, masks.shape[0], batch_size):
            masks_batch = masks[idx:idx+batch_size] 
            dmasks = masks_batch.to(inp.device).float()
            pert_inp = inp * dmasks.unsqueeze(1) + baseline * (1.0-dmasks.unsqueeze(1))
            out = model(pert_inp) ## CHNG
            mout = out.unsqueeze(-1).unsqueeze(-1)
            all_pred.append(mout.cpu())
        all_pred = torch.concat(all_pred)
        return all_pred
    
    @staticmethod
    @torch.no_grad()
    def generate_data(me, inp, baseline, catidx, masks, batch_size=32):
        fmdl = me.narrow_model(catidx, with_softmax=True)
        pred = MaskedRespGen.generate_predictions(fmdl, inp, baseline, masks, batch_size=batch_size)
        raw_pred = pred.cpu().squeeze()
        rfactor = inp.numel() ### Should be // 3 -- not changing in order not to change weights
        baseline_score = fmdl(baseline).detach().squeeze().cpu()
        label_score = fmdl(inp).detach().squeeze().cpu()

        norm = lambda x: (x - baseline_score) * rfactor                
        added_score = norm(label_score)
                
        all_pred = norm(raw_pred)

        return MaskedRespData(
            baseline_score = baseline_score,
            label_score = label_score,
            added_score = added_score,
            all_masks = masks,
            all_pred = all_pred,
            all_pred_raw = raw_pred,
            baseline = baseline
        )


class ZeroBaseline:

    def __init__(self):
        pass

    def __call__(self, inp):
        return torch.zeros(inp.shape).to(inp.device)
    
    @property
    def desc(self):
        return "Zr"


class SegSlocExpCreator(BaseSampExpCreator):
    def __init__(self, desc, seg_list=[], sq_list=[], baseline_gen=None, **kwargs):
        self.seg_list = seg_list
        self.sq_list = sq_list        
        self.kwargs = kwargs
        self.baseline_gen = baseline_gen or ZeroBaseline()
        super().__init__(desc=desc, **kwargs)


    def generate_data(self, me, inp, catidx):        
        fmdl = me.narrow_model(catidx, with_softmax=True)

        ishape = (224,224)  ## take from me
        mgen_list = []
        nmasks_list = []
        for segsize, nmasks, pprob in self.sq_list:
            if pprob < 0:
                pprob = self.get_pprob(me)                        
            mgen_list.append(SquareMaskGen(segsize, ishape, prob=pprob))
            nmasks_list.append(nmasks)
            
        for nsegs, nmasks, pprob in self.seg_list:
            if pprob < 0:
                pprob = self.get_pprob(me)
                #print(pprob)
            mgen_list.append(SegMaskGen(inp, nsegs, prob=pprob))
            nmasks_list.append(nmasks)

        print("generating masks")
        masks = MaskedRespGen.generate_masks_multi(mgen_list, nmasks_list)
        baseline = self.baseline_gen(inp)
        print("generating data")
        return MaskedRespGen.generate_data(me, inp, baseline, catidx, masks)

    
    def get_pprob(self, me):
        if 'vit_small' in me.arch:
            return 0.3            
        elif 'vit_base' in me.arch:
            return 0.2
        elif me.arch == 'resnet50':
            return 0.6            
        elif me.arch == 'densenet201':
            return 0.6
        else:
            assert False, f"Unexpected arch {me.arch}"
            
        
class RngNwSegSlocExpCreator(SegSlocExpCreator):
    def __init__(self, **kwargs):
        seg_list = seg_list = [(x, 25, -1) for x in range(20,60)] ## 60
        super().__init__(desc="RngSeg", seg_list=seg_list, **kwargs)


#########################################
#########################################


import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic


# =========================================================
# 1. SEGMENTATION NORMALIZATION
# =========================================================

def normalize_segmentation(seg):
    """
    Maps arbitrary segment ids -> 0..K-1
    """
    uniq = torch.unique(seg)
    mapping = {v.item(): i for i, v in enumerate(uniq)}

    out = torch.zeros_like(seg)
    for v, i in mapping.items():
        out[seg == v] = i

    return out, len(uniq)


# =========================================================
# 2. VIT PATCH EXTRACTION
# =========================================================

@torch.no_grad()
def extract_vit_patches(model, x):
    """
    Returns: (N_patches, D)
    Assumes:
        model.forward_features returns [CLS + patches]
    """
    feats = model.forward_features(x)

    if feats.ndim == 3:
        feats = feats[:, 1:, :]   # remove CLS
    elif feats.ndim == 2:
        raise ValueError("Unexpected ViT output format")

    return feats.squeeze(0)  # (N, D)


# =========================================================
# 3. SEGMENT EMBEDDING POOLING
# =========================================================

@torch.no_grad()
def build_segment_embeddings(seg, patch_emb, patch_size):
    """
    seg: H x W
    patch_emb: N_patches x D
    """
    H, W = seg.shape
    gh, gw = H // patch_size, W // patch_size

    seg, K = normalize_segmentation(seg)

    seg_embs = []

    for k in range(K):
        accum = []
        weights = []

        for i in range(gh):
            for j in range(gw):

                patch_mask = seg[i*patch_size:(i+1)*patch_size,
                                 j*patch_size:(j+1)*patch_size]

                w = (patch_mask == k).sum().float()

                if w > 0:
                    idx = i * gw + j
                    accum.append(patch_emb[idx])
                    weights.append(w)

        if len(accum) == 0:
            seg_embs.append(torch.zeros_like(patch_emb[0]))
            continue

        weights = torch.stack(weights)
        weights = weights / (weights.sum() + 1e-8)

        embs = torch.stack(accum)  # (n, D)        
        seg_emb = (embs.T.cpu() @ weights)

        seg_embs.append(seg_emb)

    return torch.stack(seg_embs)  # (K, D)


# =========================================================
# 4. PSD COVARIANCE FROM SIMILARITY
# =========================================================

def build_covariance(seg_emb):
    """
    Cosine similarity → PSD covariance
    """
    x = F.normalize(seg_emb, dim=1)

    S = x @ x.T

    # stabilize + PSD projection
    S = S + 0.05 * torch.eye(S.shape[0], device=S.device)

    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = torch.clamp(eigvals, min=0.0)

    Sigma = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    return Sigma


# =========================================================
# 5. CORRELATED SAMPLING
# =========================================================

VERB = 3
@torch.no_grad()
def sample_segment_latents(Sigma, n, temperature=1.0):
    K = Sigma.shape[0]

    dist = torch.distributions.MultivariateNormal(
        torch.zeros(K, device=Sigma.device),
        covariance_matrix=Sigma
    )

    z = dist.sample((n,))  # (n, K)
    #return torch.sigmoid(z / temperature)
    p = torch.distributions.Normal(0,1).cdf(z)
    mask = (p > 0.5) * 1.0
    #return mask
    #p = torch.clamp(0.5 * (p / p.mean()), min=0.2, max=0.8)
    global VERB
    if VERB > 0:
        print(n, mask.shape)
        VERB -= 1
    #mask = torch.bernoulli(p)    
    return mask


# =========================================================
# 6. MAP SEGMENTS → PIXEL MASK
# =========================================================

def segments_to_masks(seg, probs):
    """
    seg: H x W (normalized)
    probs: (n_masks, K)
    """
    H, W = seg.shape
    masks = []

    for p in probs:
        m = torch.zeros((H, W), dtype=torch.float32)

        for k in range(len(p)):
            m[seg == k] = p[k]

        masks.append(m)

    return torch.stack(masks)


# =========================================================
# 7. MAIN GENERATOR
# =========================================================

class ViTCorrelatedSegMaskGen:

    def __init__(self, model, inp, patch_size=16, temperature=1.0):
        self.model = model
        self.patch_size = patch_size
        self.temperature = temperature
        self.patch_emb = extract_vit_patches(self.model, inp)

    @torch.no_grad()
    def generate(self, seg, nmasks):

        # (1) ViT features
        patch_emb = self.patch_emb

        # (2) segment embeddings
        seg_emb = build_segment_embeddings(
            seg, patch_emb, self.patch_size
        )

        # (3) covariance
        Sigma = build_covariance(seg_emb)

        # (4) sample latent segment activations
        z = sample_segment_latents(Sigma, nmasks, self.temperature)

        # (5) convert to masks
        masks = segments_to_masks(seg, z)

        return masks



# =========================================================
# 9. PLUG INTO YOUR PIPELINE
# =========================================================

class AdaptedViTCorrelatedSegMaskGen:
    def __init__(self, vgen, seg):
        self.vgen = vgen
        self.seg = seg

    def gen_masks(self, nmasks):        
        return self.vgen.generate(self.seg, nmasks)
    



class CorrelatedSegSlocExpCreator(BaseSampExpCreator):

    SIGMA = None

    def __init__(self, desc, seg_fn, nmasks, **kwargs):
        super().__init__(desc=desc, **kwargs)
        self.seg_fn = seg_fn        
        self.nmasks = nmasks

    def generate_data(self, me, inp, catidx):

        fmdl = me.narrow_model(catidx, with_softmax=True)

        #seg_list = self.seg_fn(inp)  # <-- allows dynamic segmentation
        #gen = MultiSegmentationMaskGen(me.model)
        #masks = gen.generate(inp, seg_list, nmasks=self.nmasks)

        print("generating vgen")
        vgen = ViTCorrelatedSegMaskGen(model=me.model, inp=inp)
        
        ##### REMOVE  
        CorrelatedSegSlocExpCreator.SIGMA = build_covariance(vgen.patch_emb)
        ###
      
        print("done generating vgen")
        mgen_list = []
        nmasks_list = []
        for seg_fn in self.seg_fn:
            seg = seg_fn(inp)
            mgen = AdaptedViTCorrelatedSegMaskGen(vgen, seg)
            mgen_list.append(mgen)
            nmasks_list.append(self.nmasks)
        masks = MaskedRespGen.generate_masks_multi(mgen_list, nmasks_list)
        

        baseline = torch.zeros_like(inp)

        data = MaskedRespGen.generate_data(
            me, inp, baseline, catidx, masks
        )

        self.last_data = data ## REMOVE
        return data


def CorrelatedRngSegExpCreator():
    mshape = (224,224)
    seg_fn_list = [make_seg_fn(x, mshape) for x in range(20,60)]
    #def joint_seg_fn(inp):
    #    return [sf(inp) for sf in seg_fn_list]
    
    return CorrelatedSegSlocExpCreator(
        "CorrelatedRngSeg",
        seg_fn_list, 25)        
    

import torch
import numpy as np
from skimage.segmentation import slic


def make_square_seg_fn(segsize, mshape, efactor=4):
    """
    Returns a function seg_fn(inp=None) -> segmentation tensor (H x W)

    Each call:
        - generates a new random offset
        - builds square-like pseudo-segmentation using slic on grid image
    """

    H, W = mshape

    # precompute expanded canvas size (like your original idea)
    base_H = H * efactor
    base_W = W * efactor

    def seg_fn(inp=None):

        # ----------------------------------------------------
        # 1. random offset for tiling
        # ----------------------------------------------------
        off_h = torch.randint(0, segsize, (1,)).item()
        off_w = torch.randint(0, segsize, (1,)).item()

        # ----------------------------------------------------
        # 2. build grid-like image to force square segments
        # ----------------------------------------------------
        grid = np.zeros((base_H, base_W, 3), dtype=np.float32)

        # encode spatial structure (important for slic stability)
        for i in range(base_H):
            for j in range(base_W):
                grid[i, j, 0] = (i + off_h) // segsize
                grid[i, j, 1] = (j + off_w) // segsize
                grid[i, j, 2] = 0

        # ----------------------------------------------------
        # 3. run SLIC to stabilize square-ish regions
        # ----------------------------------------------------
        n_segments = (base_H * base_W) // (segsize * segsize)

        seg = slic(
            grid,
            n_segments=n_segments,
            compactness=1000,
            sigma=0
        )

        seg = torch.tensor(seg, dtype=torch.int32)

        # ----------------------------------------------------
        # 4. crop back to original image size
        # ----------------------------------------------------
        seg = seg[:H, :W]

        return seg

    return seg_fn



def make_seg_fn(nseg, mshape):
    """
    Returns a function seg_fn(inp) -> segmentation (H x W)

    Each call:
        - runs SLIC with random perturbation
        - produces semantically different segmentations per mask
    """

    H, W = mshape

    def seg_fn(inp):

        # ----------------------------------------------------
        # 1. use input image as SLIC base
        # ----------------------------------------------------
        if inp is None:
            base = np.zeros((H, W, 3), dtype=np.float32)
        else:
            base = inp[0].detach().cpu().numpy().transpose(1, 2, 0)

        # ----------------------------------------------------
        # 2. inject randomness (VERY IMPORTANT)
        # ----------------------------------------------------
        noise = np.random.normal(0, 0.02, base.shape).astype(np.float32)
        base_noisy = base + noise

        # ----------------------------------------------------
        # 3. run SLIC
        # ----------------------------------------------------
        seg = slic(
            base_noisy,
            n_segments=nseg,
            compactness=10,
            sigma=1,
            start_label=0
        )

        seg = torch.tensor(seg, dtype=torch.int32)

        # ----------------------------------------------------
        # 4. ensure shape consistency
        # ----------------------------------------------------
        seg = seg[:H, :W]

        return seg

    return seg_fn


###############################################
##############################################
# using attention insead of cosim

import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_attention_rollout(model, x):
    """
    Returns rollout attention matrix A of shape:
        (N_patches, N_patches)

    Assumes model exposes attention weights per layer.
    This is adapted conceptually; you may need to hook your ViT.
    """

    attn_mats = []

    def hook_fn(module, input, output):
        # output expected: (B, heads, tokens, tokens)
        attn = output[1] if isinstance(output, tuple) else output
        attn = attn.mean(dim=1)  # average heads
        attn_mats.append(attn.detach().cpu())

    hooks = []

    # ---- register hooks on all attention blocks ----
    for blk in model.blocks:
        h = blk.attn.register_forward_hook(hook_fn)
        hooks.append(h)

    _ = model.forward_features(x)

    for h in hooks:
        h.remove()

    # ---- rollout ----
    A = attn_mats[0][0]  # (N, N)

    I = torch.eye(A.shape[0])

    rollout = I + A

    for a in attn_mats[1:]:
        a = a[0]
        rollout = rollout @ (I + a)

    return rollout


# =========================================================
# SEGMENT AGGREGATION FROM ATTENTION
# =========================================================

@torch.no_grad()
def segment_from_attention(seg, attn_mat):
    """
    seg: H x W (normalized segment ids)
    attn_mat: N_patches x N_patches
    """

    H, W = seg.shape
    N = attn_mat.shape[0]

    gh, gw = H // int(N ** 0.5), W // int(N ** 0.5)

    seg = seg.clone()
    seg_ids = torch.unique(seg)

    K = len(seg_ids)

    S = torch.zeros((K, K))

    seg_map = {v.item(): i for i, v in enumerate(seg_ids)}

    for i in range(gh):
        for j in range(gw):

            p1 = i * gw + j

            patch1 = seg[i*gh:(i+1)*gh, j*gw:(j+1)*gw]

            for k1 in seg_ids:
                m1 = (patch1 == k1).float().sum()

                if m1 == 0:
                    continue

                for ii in range(gh):
                    for jj in range(gw):

                        p2 = ii * gw + jj
                        patch2 = seg[ii*gh:(ii+1)*gh, jj*gw:(jj+1)*gw]

                        for k2 in seg_ids:
                            m2 = (patch2 == k2).float().sum()

                            if m2 == 0:
                                continue

                            S[seg_map[k1.item()], seg_map[k2.item()]] += (
                                attn_mat[p1, p2] * m1 * m2
                            )

    # normalize
    S = S / (S.sum(dim=1, keepdim=True) + 1e-8)

    return S


# =========================================================
# FINAL COVARIANCE BUILDER (ATTENTION VERSION)
# =========================================================

@torch.no_grad()
def build_covariance_attention(model, inp, seg):
    """
    Returns PSD covariance matrix over segments
    using ViT attention rollout.
    """

    # 1. attention rollout
    attn = extract_attention_rollout(model, inp)

    # 2. segment affinity
    S = segment_from_attention(seg, attn)

    # 3. symmetrize (important for Gaussian sampling)
    S = 0.5 * (S + S.T)

    # 4. PSD projection (stability)
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = torch.clamp(eigvals, min=0.0)

    Sigma = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    # 5. regularization
    Sigma = Sigma + 0.05 * torch.eye(Sigma.shape[0])

    return Sigma



# =========================================================
# 8. MULTI-SEGMENTATION SUPPORT (IMPORTANT FOR YOU) --- remove -dead code
# =========================================================

class MultiSegmentationMaskGen:

    def __init__(self, model, patch_size=16, temperature=1.0):
        self.model = model
        self.patch_size = patch_size
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, inp, seg_list, nmasks):

        all_masks = []

        for seg in seg_list:

            gen = ViTCorrelatedSegMaskGen(
                self.model,
                self.patch_size,
                self.temperature
            )

            masks = gen.generate(inp, seg, nmasks)
            all_masks.append(masks)

        return torch.cat(all_masks, dim=0)

