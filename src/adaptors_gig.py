
from baselines.gig import setup
import saliency.core as saliency
import torch
import numpy as np



class IGSaliencyCreator:
    def __init__(self, nsteps=100):
        self.nsteps = nsteps

    def __call__(self, me, inp, catidx):
        orig_device = inp.device()
        inp = inp.cpu()
        device = inp.device
        model = me.model.to(device) ##.cpu()
        class_idx_str = 'class_idx_str'
        call_model_args = {class_idx_str: catidx}
        im = inp[0].transpose(0,1).transpose(1,2).cpu().numpy()

        def call_model_function(images, call_model_args=None, expected_keys=None):
            
            images = np.array(images)            
            images = np.transpose(images, (0,3,1,2))
            images = torch.tensor(images, dtype=torch.float32)            
            images = images.requires_grad_(True)
            images = images.to(device)
            
            target_class_idx =  call_model_args[class_idx_str]
            output = model(images)
            m = torch.nn.Softmax(dim=1)
            output = m(output)
            if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
                outputs = output[:,target_class_idx]
                grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
                grads = torch.movedim(grads[0], 1, 3)
                gradients = grads.detach().cpu().numpy()
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                one_hot = torch.zeros_like(output)
                one_hot[:,target_class_idx] = 1
                model.zero_grad()
                output.backward(gradient=one_hot, retain_graph=True)
                return conv_layer_outputs

        baseline = np.zeros(im.shape)
        
        integrated_gradients = saliency.IntegratedGradients()
        vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
            im, call_model_function, call_model_args, x_steps=self.nsteps, x_baseline=baseline, batch_size=20)
        ig_sal = torch.tensor(np.sum(np.abs(vanilla_integrated_gradients_mask_3d), axis=2)).unsqueeze(0).float()

        guided_ig = saliency.GuidedIG()
        guided_ig_mask_3d = guided_ig.GetMask(
        im, call_model_function, call_model_args, x_steps=self.nsteps, x_baseline=baseline, max_dist=1.0, fraction=0.5)
        gig_sal = torch.tensor(np.sum(np.abs(guided_ig_mask_3d), axis=2)).unsqueeze(0).float()
        model = me.model.to(orig_device)
        res = {f"IG_{self.nsteps}" : ig_sal, f"GIG_{self.nsteps}" : gig_sal}
        return res
