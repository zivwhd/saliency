try:
    import setup_paths
except:
    pass

from enum import Enum, auto
import torch

from pytorch_grad_cam import run_dff_on_image, GradCAM, FullGrad, LayerCAM, GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


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
    def __init__(self, methods):
        self.methods = methods

    def __call__(self, me, inp, catidx):
        res = {}
        for mthd in self.methods:   
            method = METHOD_CONV[mthd]
            cam = method(model=me.model, target_layers=[me.get_cam_target_layer()])
            targets_for_gradcam = [ClassifierOutputTarget(catidx)]

            inp.requires_grad_(True)
            sal = cam(input_tensor=inp, targets=targets_for_gradcam)
            res[mthd.name] = torch.Tensor(sal)
        return res
