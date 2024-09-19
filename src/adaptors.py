try:
    import setup_paths
except:
    pass

from enum import Enum, auto
import logging
import torch

from pytorch_grad_cam import run_dff_on_image, GradCAM, FullGrad, LayerCAM, GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import captum
import cv2


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
            method = METHOD_CONV[mthd]
            cam = method(model=me.model, target_layers=[me.get_cam_target_layer()])
            targets_for_gradcam = [ClassifierOutputTarget(catidx)]
            cinp = inp.clone().detach()
            cinp.requires_grad_(True)
            sal = cam(input_tensor=cinp, targets=targets_for_gradcam)
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
        
        attribution = attribution.squeeze().cpu().detach().to(torch.float32)
        attribution = attribution.sum(dim=0).unsqueeze(0)
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
        attribution = attribution.squeeze().cpu().detach().to(torch.float32)


        # Sum over the color channels (RGB) to get a single grayscale attribution map
        attribution = attribution.sum(axis=0).unsqueeze(0)
        return attribution.cpu()


class DixCnnSaliencyCreator:

    def __init__(self):
        pass

    def __call__(self, me, inp, catidx):
        from baselines.dix import setup

    #return DimplVitSaliencyCreator(['dix'])