
import torch, logging


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



class DimplVitSaliencyCreator:
    def __init__(self, methods = ["dix", "t-attr", "gae"], alt_model=None):
        self.methods = methods
        self.alt_model = alt_model

    def __call__(self, me, inp, catidx):
        from baselines.dix import setup
        from vit_model import ViTmodel, ViT_LRP
        from vit_model.ViT_explanation_generator import get_interpolated_values
        from saliency_utils import tensor2cv
        #from salieny_models import *
        from seg_methods_vit import blend_transformer_heatmap, transformer_attribution,  get_dix_vit, generate_map_gae

        device = inp.device

        if self.alt_model is not None:
            model = self.alt_model
        elif me.arch == "vit_small_patch16_224":
            #print("### small vit")
            model = ViTmodel.vit_small_patch16_224(pretrained=True).to(device)
        elif me.arch == "vit_base_patch16_224":
            model = ViTmodel.vit_base_patch16_224(pretrained=True).to(device)
        elif me.arch in ["resnet50"]:
            model = me.model
        else:
            raise Exception(f"unexpected arch {arch}")
        
        input_predictions = me.model(inp) ##, register_hook=True)
        #predicted_label = torch.max(input_predictions, 1).indices[0].item()
        
        res = {}

        for opr in self.methods:
            sal = self.handle_transformers(model, me.arch, opr, inp[0], catidx)
            res[f"Dimpl_{opr}"] = torch.tensor(sal).unsqueeze(0)
        return res
    

    def handle_transformers(self, model, arch, operation, image, label):
        device = image.device
        from seg_methods_vit import blend_transformer_heatmap, transformer_attribution,  get_dix_vit, generate_map_gae
        logging.debug(f"Dimpl {operation}")
        if operation == 'gae':
            #print("###", image.shape, label, image.dtype, image.device)
            gae = generate_map_gae(model, image.unsqueeze(0).to(device), label).reshape(14, 14).detach()
            im, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(image.cpu(), gae.cpu())
            return heatmap
            #img_dict.append({"image": im, "title": 'gae', 'heatmap':heatmap, "ttt":t })
        elif operation == 't-attr':
            from vit_model import ViTmodel, ViT_LRP
            if arch == "vit_base_patch16_224":
                model_tattr = ViT_LRP.vit_base_patch16_224(pretrained=True).to(device)
            elif arch == "vit_small_patch16_224":
                model_tattr = ViT_LRP.vit_small_patch16_224(pretrained=True).to(device)
            else: 
                raise Exception(f"Unexpected arch {arch}")

            t_attr = transformer_attribution(model_tattr, [], image.device, label, image.to(device), 0).detach()
            im2, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(image.cpu(), t_attr.cpu())
            return heatmap
            #img_dict.append({"image": im2, "title": 't_attr'})
        elif operation == 'dix':
            ## REVIEW - we'd like to run dix1
            dix_attribution1 = get_dix_vit(model, [], image.device, label, image.unsqueeze(0), 1)
            dix_attribution2 = get_dix_vit(model, [], image.device, label, image.unsqueeze(0), 2)
            dix_attribution=dix_attribution1 # + dix_attribution2
            im3, score, heatmap_cv, blended_img_mask, heatmap, t = blend_transformer_heatmap(image.cpu(), dix_attribution.cpu(), resize=(self.alt_model is not None) )
            #img_dict.append({"image": im3, "title": 'DIX', 'heatmap':heatmap, "ttt":t })
            return heatmap
        else:
            raise Exception(f"Unexpected operation {operation}")        

