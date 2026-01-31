try:
    import setup_paths
except:
    pass

from enum import Enum, auto
import logging,time
import torch
import numpy as np
import math

from skimage.segmentation import slic

from pytorch_grad_cam import run_dff_on_image, GradCAM, FullGrad, LayerCAM, GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import captum
import cv2
from reports import report_duration

class CMethod(Enum):
    GradCAM = auto()
    GradCAMPlusPlus = auto()
    FullGrad = auto()
    LayerCAM = auto()    
    AblationCAM = auto()

METHOD_CONV = {
    CMethod.GradCAM : GradCAM,
    CMethod.FullGrad : FullGrad,
    CMethod.LayerCAM : LayerCAM,
    CMethod.GradCAMPlusPlus : GradCAMPlusPlus,
    CMethod.AblationCAM : AblationCAM
}


class CamSaliencyCreator:
    def __init__(self, methods=list(METHOD_CONV.keys())):
        self.methods = methods

    def __call__(self, me, inp, catidx):
        res = {}
        for mthd in self.methods:   
            start_time = time.time()
            method = METHOD_CONV[mthd]
            cam = method(model=me.model, target_layers=[me.get_cam_target_layer()])
            targets_for_gradcam = [ClassifierOutputTarget(catidx)]
            cinp = inp.clone().detach()
            cinp.requires_grad_(True)
            sal = cam(input_tensor=cinp, targets=targets_for_gradcam)
            report_duration(start_time, me.arch, mthd.name, '')
            res[f"pgc_{mthd.name}"] = torch.Tensor(sal)
        return res

class CaptumCamSaliencyCreator:
    def __init__(self, methods=['GradientShap', 'LayerGradCam', 'IntegratedGradients']):
        self.methods = methods

    def __call__(self, me, inp, catidx):
        res = {}
        for method in self.methods:
            logging.debug(f"captum: {method}")
            func = getattr(self, method)
            sal = func(me, inp, catidx)
            res[f"captum{method}"] = sal
        return res

    def LayerGradCam(self, me, inp, catidx):    
        guided_gc = captum.attr.LayerGradCam(me.model, me.get_cam_target_layer())
        cinp = inp.clone().detach() 
        cinp.requires_grad_()  # Enable gradients on input
        attribution = guided_gc.attribute(cinp, target=catidx) 
        dsal = attribution.squeeze(0).squeeze(0).cpu().detach().numpy()
        sal = torch.tensor(cv2.resize(dsal, tuple(inp.shape[-2:]))).unsqueeze(0)
        return sal        
        
    def IntegratedGradients(self, me, inp, catidx):        
        ig = captum.attr.IntegratedGradients(me.model)
        cinp = inp.clone().detach() 
        cinp.requires_grad_()  # Enable gradients on input

        baseline = torch.zeros_like(cinp)  # Use a black image as the baseline

        # Compute attributions using Integrated Gradients
        attribution = ig.attribute(cinp, baseline, target=catidx, n_steps=100)

        # Convert the attribution to numpy
        
        attribution = attribution.cpu().detach().to(torch.float32)
        print("IG", attribution.shape)
        #attribution = attribution.sum(dim=0).unsqueeze(0)
        attribution = attribution.abs().sum(dim=1)
        return attribution

    def GradientShap(self, me, inp, catidx, num_baselines=20):
        
        device = inp.device
        cinp = inp.clone().detach().squeeze(0)
        gshap = captum.attr.GradientShap(me.model)

        # Create a baseline distribution by adding noise to a baseline (e.g., a black image)
        #baseline_dist = torch.cat([cinp * 0, cinp * 0 + torch.randn_like(cinp) * 0.1])
        baseline_dist = torch.randn((num_baselines,)+cinp.shape).to(device)
        #print(cinp.shape, baseline_dist.shape)
        # Perform GradientShap attribution
        cinp.requires_grad_()

        # Compute attributions using GradientShap
        attribution = gshap.attribute(cinp.unsqueeze(0), baselines=baseline_dist, target=catidx)

        # Convert the attribution to numpy for visualization
        attribution = attribution.cpu().detach().to(torch.float32)
        print("GS", attribution.shape)

        # Sum over the color channels (RGB) to get a single grayscale attribution map
        #attribution = attribution.sum(axis=0).unsqueeze(0)

        attribution = attribution.abs().sum(dim=1)
        return attribution.cpu()


class DixCnnSaliencyCreator:

    def __init__(self):
        pass

    def __call__(self, me, inp, catidx):
        from baselines.dix import setup

    #return DimplVitSaliencyCreator(['dix'])



class KernelShapSaliencyCreator:

    def __init__(self,
                 n_evals=1000,
                 n_segments=50,
                 compactness=10.0, 
                 sigma=1):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.n_evals = 1000

        
    def __call__(self, me, inp, catidx):
        sal = self.explain(me, inp, catidx,  ##
                           n_segments=self.n_segments, 
                           max_evals= self.n_evals)
        desc = f"KernelShap_{self.n_evals}_{self.n_segments}"
        return {desc : sal}


    def explain(self, me, inp: torch.Tensor, catidx: int,
                n_segments: int = 75,
                compactness: float = 10.0,
                max_evals: int = 1000,
                batch_size: int = 32,
                baseline: str = "zero",   # "zero" or "mean"
                ridge: float = 1e-6,
                seed: int = 0) -> torch.Tensor:
        """
        Short KernelSHAP (manual) over SLIC superpixels.
        Uses inp *as-is* (no unnormalization). Returns (1,H,W) tensor.

        inp: (1,C,H,W) normalized tensor. Model scalar is me.model(x)[:, catidx].
        """
        assert inp.ndim == 4 and inp.shape[0] == 1
        device = inp.device
        C, H, W = inp.shape[1], inp.shape[2], inp.shape[3]

        # SLIC on HWC float (using inp as-is)
        x_hwc = inp.detach().cpu()[0].permute(1, 2, 0).numpy().astype(np.float32)
        segments = slic(x_hwc, n_segments=n_segments, compactness=compactness,
                        start_label=0, channel_axis=-1)
        K = int(segments.max()) + 1
        sp_masks = [(segments == k) for k in range(K)]

        # Baseline (in same space as inp)
        if baseline == "zero":
            base = np.zeros((H, W, C), dtype=np.float32)
        elif baseline == "mean":
            base_color = x_hwc.reshape(-1, C).mean(axis=0)
            base = np.broadcast_to(base_color, (H, W, C)).copy()
        else:
            raise ValueError("baseline must be 'zero' or 'mean'")

        # Shapley kernel weight
        def w_s(s: int) -> float:
            if s == 0 or s == K:
                return 1e6
            return (K - 1) / (math.comb(K, s) * s * (K - s))

        rng = np.random.default_rng(seed)

        # Sample coalitions Z (M,K); include empty & full
        M = max(2, int(max_evals))
        Z = np.zeros((M, K), dtype=np.int8)
        Z[0, :] = 0
        Z[1, :] = 1
        for i in range(2, M):
            s = int(rng.integers(1, K))
            idx = rng.choice(K, size=s, replace=False)
            Z[i, idx] = 1

        sizes = Z.sum(axis=1)
        Wgt = np.array([w_s(int(s)) for s in sizes], dtype=np.float64)

        @torch.no_grad()
        def eval_batch(zb: np.ndarray) -> np.ndarray:
            B = zb.shape[0]
            imgs = np.empty((B, H, W, C), dtype=np.float32)
            for bi in range(B):
                im = base.copy()
                for k in np.nonzero(zb[bi])[0]:
                    m = sp_masks[int(k)]
                    im[m] = x_hwc[m]
                imgs[bi] = im
            t = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device=device, dtype=inp.dtype)
            y = me.model(t)[:, catidx]
            return y.detach().cpu().numpy().astype(np.float64)

        y = np.empty((M,), dtype=np.float64)
        for s in range(0, M, batch_size):
            e = min(M, s + batch_size)
            y[s:e] = eval_batch(Z[s:e])

        # Weighted linear regression: y ~ b0 + Z @ phi
        X = np.concatenate([np.ones((M, 1), dtype=np.float64), Z.astype(np.float64)], axis=1)
        XT_W = X.T * Wgt
        A = XT_W @ X
        b = XT_W @ y
        I = np.eye(K + 1, dtype=np.float64); I[0, 0] = 0.0
        beta = np.linalg.solve(A + ridge * I, b)
        phi = beta[1:]  # (K,)

        # Broadcast to pixels -> (1,H,W)
        pix = np.zeros((H, W), dtype=np.float32)
        for k in range(K):
            pix[sp_masks[k]] = float(phi[k])
        return torch.from_numpy(pix).unsqueeze(0).to(device)
