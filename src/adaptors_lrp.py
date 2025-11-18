from functools import partial
import torch, logging, time
from reports import report_duration

def model_handler(pretrained=False,size = "tiny" , hooks = False,  **kwargs):
        from models.model_visualizations import deit_tiny_patch16_224 as vit_LRP
        from models.model_visualizations import deit_base_patch16_224 as vit_LRP_base
        #from models.model_visualizations import deit_small_patch16_224 as vit_LRP_small        
        from models.model import  lrp_vit_small_patch16_224 as vit_LRP_small        
        from modules.layers_ours import GELU, CustomLRPLayerNorm
        if size == 'base':
            return vit_LRP_base(

                    num_classes     = 1000,
                    pretrained      = pretrained,
                    activation      = GELU(),
                    layer_norm      = partial(CustomLRPLayerNorm, eps=1e-6),
                    last_norm       = CustomLRPLayerNorm
            )
        elif size == 'small':
            print("###### small")
            return vit_LRP_small(
                    num_classes     = 1000,
                    pretrained      = pretrained,
                    activation      = GELU(),
                    layer_norm      = partial(CustomLRPLayerNorm, eps=1e-6),
                    last_norm       = CustomLRPLayerNorm
            )
        return vit_LRP(
                   num_classes     = 1000,
                   pretrained      = pretrained,
                   activation      = GELU(),
                   layer_norm      = partial(CustomLRPLayerNorm, eps=1e-6),
                   last_norm       = CustomLRPLayerNorm

            )


class DepPreAwareLRPSaliencyCreator:
    def __init__(self):
        pass

    def __call__(self, me, inp, catidx, method='custom_lrp'):

        from baselines.PreAwareLRP import setup
        from ViT_explanation_generator import LRP, LRP_RAP

        model_name = me.arch
        if 'vit_base' in model_name:
             model_size = 'base'
        if 'vit_small' in model_name:
             model_size = 'small'
            
        model = model_handler(pretrained = True,
                      size       = model_size,
                      hooks      = True)


        model.cuda()
        model.eval()


        prop_rules = {
            'epsilon_rule' : False,
            'conv_gamma_rule'  :  False,
            'linear_gamma_rule'  :False,
            "linear_alpha_rule" : 1,
            "default_op"   : False,
        }

        DEFAULT_GAMMA_LINEAR = 0.05
        DEFAULT_GAMMA_CONV = 100
        DEFAULT_ALPHA_LINEAR = 1
        linear_gamma_rule = DEFAULT_GAMMA_LINEAR
        conv_gamma_rule = DEFAULT_GAMMA_CONV
        linear_alpha_rule = 1
        ## from config.py
        prop_rules['epsilon_rule'] = True if 'epsilon_rule' in method or 'GammaLinear' in method else False
        prop_rules['conv_gamma_rule']   = conv_gamma_rule if 'gammaConv' in method else False
        prop_rules['linear_gamma_rule']  = linear_gamma_rule if 'GammaLinear' in method else False
        prop_rules["linear_alpha_rule"]   = linear_alpha_rule
        prop_rules["default_op"]   = True if 'semiGammaLinear' in method else False
        conv_prop_rule = method.split("_")[-1]
        
        attribution_generator = LRP(model)
        ## from configs
        conv_prop_rule = method.split("_")[-1]

        attr = attribution_generator.generate_LRP(inp.cuda(), 
                                                  prop_rules = prop_rules, 
                                                  conv_prop_rule=conv_prop_rule,
                                                  method=method, cp_rule=True,  
                                                  index=torch.tensor([catidx]))
        return attr


class ArgsObj:
     pass

class EPreAwareLRPSaliencyCreator:
    def __init__(self, pos=True, method=None):
        if method:
            self.method = method
        elif pos:
             self.method = 'full_lrp_GammaLinear_POS_ENC_gammaConv'
        else:
             self.method = 'full_lrp_GammaLinear_gammaConv'

    def __call__(self, me, inp, catidx):
        from baselines.PreAwareLRP import setup
        import config
        from models.model_handler import simp_model_env
        from ViT_explanation_generator import Baselines, LRP

        args = ArgsObj()
        args.method = self.method
        args.variant = 'basic'
        args.data_path=None
        args.data_set = None
        args.grid = False
        args.nb_classes = 1000

        config.get_config(args, skip_further_testing = True)
        config.set_components_custom_lrp(args, gridSearch= args.grid)
        if 'base' in me.arch:
            args.model_components['size'] = 'base'
        elif 'small' in me.arch:
             args.model_components['size'] = 'small'

        model = simp_model_env(args)
        model.to(inp.device)
        model.eval()
        assert 'full_lrp' in args.method
        lrp = LRP(model)
        Res  = lrp.generate_LRP(inp.cuda(), method=args.method,  cp_rule = args.cp_rule, 
                                prop_rules = args.prop_rules,  
                                conv_prop_rule = args.conv_prop_rule, 
                                index=torch.tensor([catidx])).reshape(1, 224, 224).detach().cpu()

        Org = Res
        Res = -Res
        if (Res.max() - Res.min()) > 0:
            Res = (Res-Res.min()) / (Res.max() - Res.min())
        return { self.method : Res, f'{self.method}_o' : Org}
