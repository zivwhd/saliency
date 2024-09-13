
import torch


class AttrVitSaliencyCreator:
    def __init__(self):
        pass

    def __call__(self, me, inp, catidx):

        from baselines.AttrViT import setup
        from ViT_LRP import vit_base_patch16_224, vit_small_patch16_224 
        from ViT_explanation_generator import LRP

        if me.arch == 'vit_small_patch16_224':
            model_type = vit_small_patch16_224
        elif me.arch == 'vit_base_patch16_224':
            model_type = vit_base_patch16_224
        else:
            raise Exception(f"Unexpected architecture {me.arch}")            
        
        model = model_type(pretrained=True).cuda()
        
        model.eval()
        attribution_generator = LRP(model)
        transformer_attribution = attribution_generator.generate_LRP(inp, method="transformer_attribution", index=catidx).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().unsqueeze(0) ##.numpy()

        return dict(TAttr=transformer_attribution)



class DIVitSaliencyCreator:
    def __init__(self):
        pass

    def __call__(self, me, inp, catidx):
        from baselines.dix import setup
        from vit_model import ViTmodel
        from vit_model.ViT_explanation_generator import get_interpolated_values
        device = inp.device
        model = ViTmodel.vit_small_patch16_224(pretrained=True).to(device)
        input_predictions = model(inp)        
        predicted_label = torch.max(input_predictions, 1).indices[0].item()

    
