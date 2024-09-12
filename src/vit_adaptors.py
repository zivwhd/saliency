
from baselines.AttrViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.AttrViT.ViT_explanation_generator import LRP

model = vit_LRP(pretrained=True).cuda()
model.eval()
attribution_generator = LRP(model)
transformer_attribution = attribution_generator.generate_LRP(inp, method="transformer_attribution", index=topidx).detach()
transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu() ##.numpy()

class AttrVitSaliencyCreator:
    def __init__(self):
        pass

    def __call__(self, me, inp, catidx):

        if me.arch != 'vit_small_patch16_224':
            raise Exception(f"Unexpected architecture {me.arch}")            
        
        model = vit_LRP(pretrained=True).cuda()
        model.eval()
        attribution_generator = LRP(model)
        transformer_attribution = attribution_generator.generate_LRP(inp, method="transformer_attribution", index=topidx).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().unsqueeze(0) ##.numpy()

        return dict(TAttr=transformer_attribution)
