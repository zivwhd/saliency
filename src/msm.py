import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import logging, time, pickle
from skimage.segmentation import slic, mark_boundaries
import numpy as np

from collections import defaultdict
from scipy.sparse.linalg import cg, gmres, lsqr
import functools
import math

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

tqdm = lambda x: x



class SegmentAssignment:

    def __init__(self, inp, n_segments, prob=0.5, compactness=10, sigma=1):
        base = inp[0].cpu().numpy().transpose(1,2,0)
        self.segments = torch.tensor(slic(base,n_segments=n_segments,compactness=compactness,sigma=sigma), dtype=torch.int32)        
        self.nelm = torch.unique(self.segments).numel()
        self.mshape = inp.shape
        self.prob = prob
        self.centroids = self.get_centroids()
    
    def get_centroids(self):
        shape = self.mshape[-2:]
        print("centroids", shape)
        xx = torch.zeros(shape) + torch.arange(shape[1]).unsqueeze(0)
        yy = torch.zeros(shape) + torch.arange(shape[0]).unsqueeze(1)
        positions = torch.arange(shape[1]).unsqueeze(0) + torch.arange(shape[0]).unsqueeze(1) * shape[1]
        segid = (torch.arange(self.nelm)+1).unsqueeze(1)
        segmasks = self.segments.flatten().unsqueeze(0) == segid
        
        cx = (segmasks * xx.flatten().unsqueeze(0)).sum(dim=1) / (segmasks.sum(dim=1))
        cx = cx.round().long()
        cy = (segmasks * yy.flatten().unsqueeze(0)).sum(dim=1) / (segmasks.sum(dim=1))
        cy = cy.round().long()
        centroids = cx + cy * shape[1]        
        #centroids = (segmasks * positions.flatten().unsqueeze(0)).sum(dim=1) / (segmasks.sum(dim=1) + 0.0001)
        #centroids = centroids.round().long()
        return centroids

    def get_random_assignments(self, nsamps):
        return torch.rand((nsamps, self.nelm))
    
    def assignments_to_masks(self, assignments):        
        #print("Generating segments mask")
        nsamps = assignments.shape[0]
        step = self.nelm
        nelm = step * nsamps
        #rnd = torch.rand(2+nelm)
        stt = []
        for idx in range(nsamps):
            wseg = self.segments + step * idx - 1
            stt.append(wseg)
        parts = torch.stack(stt)
        #print(self.nelm, rnd.shape, self.segments.shape, len(self.segments.unique()), len(parts.unique()))
        return assignments.flatten()[parts.view(-1)].view(parts.shape)
    
    def masks_to_assignments(self, masks):
        
        shape = self.mshape[-2:]
        nsamps = masks.shape[0]
        assignments = []
        for mask in masks:
            assignments.append(mask.flatten()[self.centroids])
        return torch.stack(assignments)
    
    def masks_to_soft_assignments(self, masks):
        segid = (torch.arange(self.nelm)+1).unsqueeze(1)
        pt = self.segments.flatten().unsqueeze(0) == segid
        pt = pt / pt.sum(dim=1, keepdim=True)
        assignments = torch.stack([(pt  *   masks[idx].flatten().unsqueeze(0)).sum(dim=1) for idx in range(masks.shape[0])])
        return assignments

        

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


class SqMaskGen:

    def __init__(self, segsize, mshape, efactor=4, prob=0.5):        
        base = np.zeros((mshape[0] * efactor, mshape[1] * efactor ,3), np.int32)
        n_segments = base.shape[0] * base.shape[1] / (segsize * segsize)
        self.setsize = n_segments
        self.segments = torch.tensor(slic(base,n_segments=n_segments,compactness=1000,sigma=1), dtype=torch.int32)
        self.nelm = torch.unique(self.segments).numel()
        self.prob = prob
        self.mshape = mshape
    
    def gen_masks(self, nmasks):        
        masks = (self.gen_masks_cont(nmasks) < self.prob)        
        return masks
    
    def gen_masks_cont(self, nmasks):
        
        height, width = self.mshape        
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


class MarginalStructuralExplanation:
    
    def __init__(self, nsegs=500, nmasks=None, patchsize=56):
        self.nsegs = nsegs
        self.nmasks = nmasks
        self.patchsize = patchsize

    
    def get_segments(self, inp, nsegs):
        return SegmentAssignment(inp, nsegs)

    @torch.no_grad()
    def generate_mask_pred(self, me, inp, catidx, nmasks, patchsize=40, prob=0.5, batch_size=64):
        mgen = SqMaskGen(patchsize, inp.shape[-2:])
        model = me.narrow_model(catidx, with_softmax=True)        
        total = 0
        pred_list = []
        mask_list = []
        print(f"## generating {nmasks} masks and responses")
        while total < nmasks:
            cbs = min(batch_size, nmasks - total)
            masks = mgen.gen_masks_cont(cbs)
            is_valid = (masks.flatten(start_dim=1).sum(dim=1) > 0)
            if (not any(is_valid)):
                continue
            masks = masks[is_valid]
            dmasks = masks.to(inp.device).float()
            pert_inp = inp * dmasks.unsqueeze(1) ##  baseline * (1.0-dmasks.unsqueeze(1))
            out = model(pert_inp) ## CHNG
            mout = out.clone().detach().cpu()
            pred_list.append(mout)
            mask_list.append(masks)
            total += int(is_valid.sum())

        masks = torch.concat(mask_list)        
        pred = torch.concat(pred_list)        
        return masks, pred

    @torch.no_grad()
    def generate_assignment_pred(self, me, inp, catidx, sga, nmasks, prob=0.5, 
                                 patchsize=0, batch_size=64):
        
        if patchsize:
            mgen = SqMaskGen(patchsize, inp.shape[-2:])
        else:
            mgen = None

        model = me.narrow_model(catidx, with_softmax=True)        
        total = 0
        
        assignments_list = []
        mask_list = []
        pred_list = []
        base_list = []
        print(f"## generating {nmasks} masks and responses")
        while total < nmasks:
            cbs = min(batch_size, nmasks - total)

            if mgen:
                base_masks = mgen.gen_masks_cont(cbs)
                assignments = (sga.masks_to_assignments(base_masks) < prob)
            else:
                assignments = (sga.get_random_assignments(cbs) < prob)
                base_masks = torch.zeros((cbs,)+inp.shape[-2:] )
            masks = sga.assignments_to_masks(assignments)
            is_valid = (masks.flatten(start_dim=1).sum(dim=1) > 0)

            if (not any(is_valid)):
                continue

            masks = masks[ is_valid ]
            assignments = assignments [ is_valid ]
            base_masks = base_masks[is_valid]

            dmasks = masks.to(inp.device).float()
            pert_inp = inp * dmasks.unsqueeze(1) ##  baseline * (1.0-dmasks.unsqueeze(1))
            #print("##", pert_inp.shape)
            out = model(pert_inp) ## CHNG
            #mout = out.unsqueeze(-1).unsqueeze(-1).clone().detach().cpu()
            mout = out.clone().detach().cpu()
            pred_list.append(mout)
            mask_list.append(masks)
            base_list.append(base_masks)
            assignments_list.append(assignments)
            total += int(is_valid.sum())
            #print(f"generated {total} / {nmasks}")

        assignments = torch.concat(assignments_list)
        masks = torch.concat(mask_list)
        base = torch.concat(base_list)
        pred = torch.concat(pred_list)
        
        return assignments, base, masks, pred


    def solve(self, sga, assignments, pred, with_bias=True, alpha=0, l1wt=0):
        
        print(f"## fitting: bias={with_bias}; alpha={alpha}")

        Y = pred
        X = assignments * 1.0
        if with_bias:
            X = torch.concat([torch.ones(X.shape[0],1), X], dim=1)

        gmdl = sm.GLM(Y.numpy(), X.numpy(), family=sm.families.Binomial())
        
        if alpha:
            results = gmdl.fit_regularized(
                method='elastic_net',       # Use 'l1' method for L2, L1, or ElasticNet regularization
                alpha=alpha,    # The L2 penalty strength (lambda)
                L1_wt=l1wt          # L1_wt = 0.0 specifies pure L2 (Ridge) penalty
                )
        else:
            results = gmdl.fit()        
        attr = torch.tensor(results.params, dtype=torch.float32)

        if with_bias:
            attr = attr[1:]

        #print(attr.shape)
        exp = sga.assignments_to_masks(attr.unsqueeze(0))
        return attr, exp


    def gaussian_blur_torch(self, img_tensor, radius):
        # 1. Define Kernel Size and Sigma
        # The standard deviation (sigma) is often related to the radius/kernel size.
        # Common convention: sigma = radius or sigma = (kernel_size - 1) / (2 * 2.35)
        kernel_size = 2 * radius + 1
        sigma = radius / 2.0  # A common heuristic for sigma based on radius
        # Ensure the input tensor has the expected (N, C, H, W) shape for 2D convolution
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)        
        N, C, H, W = img_tensor.shape
        # 2. Generate the 1D Gaussian kernel
        # Create a 1D sequence of numbers from -radius to +radius
        coords = torch.arange(kernel_size, dtype=torch.float32) - radius        
        # Calculate the 1D Gaussian values
        g_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        g_1d /= g_1d.sum() # Normalize the 1D kernel
        # 3. Generate the 2D separable Gaussian kernel
        # Outer product gives the 2D kernel (g_2d = g_1d * g_1d.T)
        g_2d = g_1d.unsqueeze(0) * g_1d.unsqueeze(1)        
        # Reshape the kernel for torch.nn.functional.conv2d
        # Shape must be: (out_channels, in_channels/groups, kernel_height, kernel_width)
        # We use a depthwise convolution (groups=C, in_channels=C, out_channels=C)
        kernel = g_2d.repeat(C, 1, 1, 1) # C kernels, each 1xkernel_size x kernel_size        
        # Move kernel to the same device as the input tensor
        kernel = kernel.to(img_tensor.device)
        # 4. Perform Convolution
        # Use padding to keep the output size the same as the input size
        blurred_tensor = F.conv2d(
            input=img_tensor, 
            weight=kernel, 
            padding=radius, # Padding is half of the kernel size
            groups=C        # Use grouped convolution for separate color channels (depthwise)
        )        
        # Return to the original shape if it was (C, H, W)
        if N == 1 and img_tensor.shape[0] == 1:
            return blurred_tensor.squeeze(0)            
        return blurred_tensor




class SoftMsmExpCreator(MarginalStructuralExplanation):
    def __init__(self, desc,
                 patches={32:600, 56:600},
                 prob=0.5,
                 nsegs=list(range(50,200,20)),
                 alphas=[0,0.1,0.5,1],
                 l1wt=0):

        self.desc = desc
        self.patches = patches
        self.prob=prob
        self.nsegs = nsegs
        self.alphas = alphas
        self.l1wt = l1wt

    def __call__(self, me, inp, catidx):
        shape = inp.shape[-2:]
        res = {}
        mask_list, pred_list = [], []
        mse = MarginalStructuralExplanation()
        for patchsize, nmasks in self.patches.items():
            masks, pred = mse.generate_mask_pred(me,inp,catidx, nmasks=nmasks,patchsize=patchsize,prob=self.prob)
            mask_list.append(masks)
            pred_list.append(pred)

        masks = torch.concat(mask_list)
        pred = torch.concat(pred_list)

        for alpha in self.alphas:
            print(f"creating for alpha={alpha}")
            exps = []
            for idx in self.nsegs:    
                sga = SegmentAssignment(inp, idx)
                print(idx, sga.nelm)
                assignment = sga.masks_to_soft_assignments(masks)
                attr, exp = mse.solve(sga, assignment, pred, alpha=alpha, l1wt=self.l1wt)
                exps.append(exp)
            mexp = torch.stack(exps).mean(dim=0)
            desc = f"SMSM_{self.desc}"
            if alpha:
                desc += f"a{alpha}l{self.l1wt}"
            res[desc] = mexp
        return res

class MsmExpCreator(MarginalStructuralExplanation):
    
    def __init__(self, nsegs=100, nmasks=1500, 
                 prob=0.5, patchsize=0,
                 alphas=[0, 0.05, 0.1, 0.5, 1], 
                 l1wt=0,
                 blur_radius=[0]):
        
        self.nsegs = nsegs
        self.prob = prob
        self.nmasks = nmasks
        self.alphas = alphas
        self.blur_radius = blur_radius
        self.patchsize = patchsize
        self.l1wt = l1wt


    def __call__(self, me, inp, catidx):
        shape = inp.shape[-2:]
        res = {}
        sga = self.get_segments(inp, self.nsegs)
        print(f"actual nseg: {sga.nelm}")
        assignments, _base, _masks, pred = self.generate_assignment_pred(me, inp, catidx, sga, self.nmasks, 
                                                                         prob=self.prob, patchsize=self.patchsize)
        
        for alpha in self.alphas:
            _attr, exp = self.solve(sga, assignments, pred, with_bias=True, alpha=alpha, l1wt=self.l1wt)
            for br in self.blur_radius:
                texp = exp
                if br:
                    radius = int(math.sqrt(shape[0]*shape[1]/sga.nelm)*br)
                    texp = self.gaussian_blur_torch(exp, radius)

                desc = f"Msm{self.nsegs}x{self.nmasks}x{self.prob}"
                if self.patchsize:
                    desc += f"s{self.patchsize}"
                if alpha:
                    desc += f"a{alpha}l{self.l1wt}"
                if br:
                    desc += f"br{br}"
                res[desc] = texp

        return res