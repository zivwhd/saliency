
import numpy as np
from skimage.segmentation import slic
from sklearn.linear_model import Lasso, LinearRegression
from scipy.special import binom
import torch
import torch.nn.functional as F

from scipy.optimize import lsq_linear

MMM = dict()

def _generate_paired_coalitions(M, n_samples, pair=False):
    # We exclude k=0 and k=M, so we sample k in [1, M-1]
    ks = np.arange(1, M)
    #ks = np.arange(0, M+1)
    # Calculate weights for the sampling distribution
    # Note: These are for picking WHICH k to sample
    sampling_weights = np.array([1.0 / (k * (M - k) * binom(M, k)) for k in ks])
    sampling_weights /= sampling_weights.sum()

    if pair:
        n_pairs = n_samples // 2
    else: 
        n_pairs = n_samples
    pair_counts = np.random.multinomial(n_pairs, sampling_weights)
    
    zs = []
    weights = []
    for k in range(1,M):
        # Calculate the kernel weight for this specific k
        # This is what goes into the W matrix in the solver
        #w_k = (M - 1) / (binom(M, k) * k * (M - k))
        w_k = 1 / (k * (M - k))

        #w_k = 1 / (k * (M - k))
        #w_k = w_k / w_k.sum()
        count = (n_samples + (M-2)) // (M-1)
        for _ in range(count):
            z = np.zeros(M)
            idx = np.random.choice(M, k, replace=False)
            z[idx] = 1
            
            # Add pair
            zs.append(z)
            if pair:
                zs.append(1 - z)
            
            # Both z and its complement (1-z) share the same kernel weight
            # because binom(M, k) == binom(M, M-k) and k(M-k) == (M-k)k
            if pair:
                weights.extend([w_k, w_k])
            else:
                weights.extend([w_k])
            
    if True:
        weights.extend([1e6])
        zs.append(np.ones(M))
        

    return np.array(zs), np.array(weights)

def solve_constrained_regression(zs, y, weights, delta):
    """
    zs: (N, M) binary coalition matrix
    y: (N,) centered model outputs (f(z) - f(baseline))
    weights: (N,) kernel weights
    delta: (1,) the total difference (f(image) - f(baseline))
    """
    N, M = zs.shape
    
    # W is the diagonal weight matrix
    # We use sqrt(W) for the internal projection to improve conditioning
    W = np.diag(weights)
    
    # We solve the KKT system:
    # [ 2 * X^T * W * X   1 ] [ phi ]   [ 2 * X^T * W * y ]
    # [        1^T        0 ] [ L   ] = [      delta      ]
    
    XTWX = zs.T @ W @ zs
    XTWy = zs.T @ (weights * y)
    
    # Build the block matrix
    KKT_top = np.hstack([2 * XTWX, np.ones((M, 1))])
    KKT_bottom = np.hstack([np.ones((1, M)), [[0]]])
    KKT_matrix = np.vstack([KKT_top, KKT_bottom])
    
    rhs = np.append(2 * XTWy, delta)
    
    # Use pseudo-inverse or lstsq if the matrix is singular (too few samples)
    try:
        solution = np.linalg.solve(KKT_matrix, rhs)
    except np.linalg.LinAlgError:
        # Fallback for underdetermined systems
        solution = np.linalg.lstsq(KKT_matrix, rhs, rcond=None)[0]
        
    return solution[:M] # These are your SHAP values (phi)



class SimpleKernelSHAPCreator:
    def __init__(self,
                 n_samples=1000,
                 n_segments=50,
                  compactness=30, sigma=3):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.n_samples = n_samples
        self.segments = None

        
    def __call__(self, me, inp, catidx):
        sal =  self.explain(me, inp, catidx)
        return sal
    
    def _get_shapley_kernel_weight(self, subset_size, total_features):
        """Mathematical Shapley Kernel: Weight = (M-1) / ( (M choose |S|) * |S| * (M-|S|) )"""
        if subset_size == 0 or subset_size == total_features:
            assert False ," unexpected subset_size"
            return 1e6  # Mathematically infinite, represented by a large constant
        
        weight = (total_features - 1) / (binom(total_features, subset_size) * subset_size * (total_features - subset_size))
        return weight

    def explain(self, me, inp, catidx, start_label=0):
        # 1. Segment the image into 'features' (superpixels)
        img_np = inp[0].detach().cpu().permute(1, 2, 0).numpy()
        segments = slic(
            img_np, 
            n_segments=self.n_segments, 
            compactness=self.compactness, 
            sigma=self.sigma, 
            start_label=start_label
        )

        self.segments = segments
        unique_segments = np.unique(segments)
        M = len(unique_segments)
        
        # Define background (mean color of image)    
        mean_baseline = inp.mean(dim=(2, 3), keepdim=True).expand_as(inp)
        zs, weights = _generate_paired_coalitions(M, self.n_samples) # (N, M)
        n_samples = weights.shape[0]
        #print("### ", n_samples, self.n_samples)

        def model_softmax(input_tensor):
        # Captum may pass a batch of perturbations; we return probabilities for each
            with torch.no_grad():
                logits = me.model(input_tensor)
                #return logits
                return F.softmax(logits, dim=1)


        orig_pred = model_softmax(inp).cpu()
        baseline_pred = model_softmax(mean_baseline.to(inp.device)).cpu()
        target_diff = (orig_pred-baseline_pred)[0, catidx]
        baseline_val = baseline_pred[0, catidx].item()

        # 3. Get model predictions for masked images
        masked_predictions = []
        #print(segments.shape, inp.shape, mean_baseline.shape)
        
        for z in zs:
            # Create a masked image based on coalition z
            mask = torch.ones(inp.shape[-2:])
            for i, active in enumerate(z):
                if not active:
                    mask = mask * (segments != unique_segments[i])                    
            #print(mask.sum()/mask.numel())
            mask4d = mask.unsqueeze(0).unsqueeze(0).to(inp.device)
            pert = inp * mask4d + mean_baseline * (1 - mask4d)
            #pert = (inp * mask.unsqueeze(0).unsqueeze(0).to(inp.device) +
            #        mean_baseline * (1.0-mask.unsqueeze(0).unsqueeze(0).to(inp.device))
            #)
            pred = model_softmax(pert)
            #print("###", pred.shape)
            masked_predictions.append(pred.cpu()[0,catidx] - baseline_val)
                
        n_samples = weights.shape[0]
        y = np.array(masked_predictions).reshape(n_samples, -1)
        zz = (zs * 1.0) ## zs is already numpy

        # 4. Weighted Linear Regression (Axiom of Efficiency)
        # We solve: y = X * phi
        # We use a Linear Model where input is the binary matrix 'zs'
        

        if True:
            model_reg = LinearRegression(fit_intercept=False)
            #model_reg = Lasso(fit_intercept=False, alpha=0.000 )
            model_reg.fit(zs, y, sample_weight=weights)
            print("###", model_reg.coef_) 
            coef = model_reg.coef_[0]
            
        else:
            coef = solve_constrained_regression(zz, y[:,0], weights, target_diff.numpy())
            #coef = solve_with_scipy(zz, y, weights, target_diff)

        #print(coef)

        total_shap = np.sum(coef)
        actual_delta = target_diff.item()

        #print(f"Sum of SHAPs: {total_shap:.6f}")
        #print(f"Actual Delta: {actual_delta:.6f}")
        #print(f"Difference: {total_shap - actual_delta:.6f}")

        sal = torch.zeros(inp.shape[-2:])
        for idx in range(M):
            sal = sal + coef[idx] * (segments == unique_segments[idx])                    
        res = {"SimpleKSHAP":sal.cpu().unsqueeze(0)}
        return res
    

# Usage Example:
# explainer = SimpleKernelSHAP(model.predict, n_segments=40, n_samples=500)
# shap_values, segments = explainer.explain(my_image)