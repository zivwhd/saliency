import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic
from captum.attr import KernelShap

def slic_kernel_shap(model, inp, catidx, n_segments=50, compactness=30, sigma=3, start_label=0, n_samples=1000):
    """
    Corrected version handling CUDA, batch dimensions, and 4D tensor constraints.
    """
    model.eval()
    
    # 1. Softmax Wrapper
    def model_softmax(input_tensor):
        # Captum may pass a batch of perturbations; we return probabilities for each
        with torch.no_grad():
            logits = model(input_tensor)
            return F.softmax(logits, dim=1)

    # 2. SLIC Segmentation 
    # SLIC needs HWC. We take the first image in the batch [0]
    img_np = inp[0].detach().cpu().permute(1, 2, 0).numpy()
    segments = slic(
        img_np, 
        n_segments=n_segments, 
        compactness=compactness, 
        sigma=sigma, 
        start_label=start_label
    )
    
    # 3. Create Feature Mask (1, 1, H, W)
    # This will broadcast across channels and batch automatically
    feature_mask = torch.from_numpy(segments).to(inp.device).long()
    feature_mask = feature_mask.unsqueeze(0).unsqueeze(0) # Becomes (1, 1, H, W)

    # 4. Mean Baseline (1, C, H, W)
    # Calculate mean per channel across the spatial dimensions
    mean_baseline = inp.mean(dim=(2, 3), keepdim=True).expand_as(inp)
    
    # 5. Kernel SHAP Attribution
    ks = KernelShap(model_softmax)
    
    # REMOVED .unsqueeze(0) from the call below because inp/mask/baseline are already 4D
    attributions = ks.attribute(
        inp, 
        target=catidx, 
        feature_mask=feature_mask,
        baselines=mean_baseline,
        n_samples=n_samples
    )
    
    return attributions.sum(dim=1).cpu()

class CaptumKernelShapSaliencyCreator:

    def __init__(self,
                 n_samples=1000,
                 n_segments=50,
                  compactness=30, sigma=3):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.n_samples = n_samples

        
    def __call__(self, me, inp, catidx):
        sal = slic_kernel_shap(me.model, inp, catidx, 
                               n_segments=self.n_segments, compactness=self.compactness, 
                               sigma=self.sigma, n_samples=self.n_samples)
        return {
            f'CapKernelShap{self.n_segments}_{self.n_samples}' : sal
        }