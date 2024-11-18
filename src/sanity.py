from lcpe import *
import copy

import torch
from timm import create_model




def randomize_layer(me, layer_index):    
    modified = False
    conv_layers = [x for x in me.model.named_parameters() if "conv" in x[0]]
    param = conv_layers[layer_index][1]
    param.data = torch.randn_like(param.data) 

class SanityCreator:
    def __init__(self, nmasks=500, c_magnitude=0.01):
        self.lsc = CompExpCreator(nmasks=nmasks, segsize=40, c_mask_completeness=1.0, c_magnitude=c_magnitude, 
                                  c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=None)
        
        self.lsca = CompExpCreator(nmasks=500, segsize=40, c_mask_completeness=1.0, c_magnitude=0.01, 
                                  c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=None)

    def __call__(self, me, inp, catidx, data=None):

        orig_model = me.model
        res = {}
        
        conv_layers = [x for x in me.model.named_parameters() if "conv" in x[0]]
        nlayers = len(conv_layers)
        
        res["Base"] = self.lsc.explain(me, inp, catidx).cpu()

        for idx in range(nlayers):
            layer_id = nlayers - idx
            try:
                me.model = copy.deepcopy(orig_model)

                randomize_layer(me, idx)                
                res[f"Rnd_{layer_id}"] = self.lsca.explain(me, inp, catidx).cpu()
                for lidx in range(idx, nlayers):
                    randomize_layer(me, lidx)
                res[f"Csc_{layer_id}"] = self.lsca.explain(me, inp, catidx).cpu()
            finally:
                me.model = orig_model

        return res


