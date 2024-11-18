from lcpe import *
import copy

import torch
from timm import create_model




def randomize_layer(me, layer_index):    
    modified = False
    for name, param in me.model.named_parameters():
        if (f".denselayer{layer_index}." not in name):
            continue
        modified = True
        logging.info(f"randomizing layer weights {name}")
        if 'bias' in name:
            random_weights = torch.randn_like(param.data) * 3 + 0.0
        else:
            random_weights = torch.randn_like(param.data) * 3 + 1.0
        #random_weights = torch.randn_like(param.data) * param.data.std() * 4 + param.data.mean()        
        # Replace the weights
        param.data = random_weights
    assert modified

class SanityCreator:
    def __init__(self, nmasks=500, c_magnitude=0.01):
        self.lsc = CompExpCreator(nmasks=nmasks, segsize=40, c_mask_completeness=1.0, c_magnitude=c_magnitude, 
                                  c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=None)

    def __call__(self, me, inp, catidx, data=None):

        orig_model = me.model
        res = {}
        
        res["Base"] = self.lsc.explain(me, inp, catidx).cpu()

        for idx in range (1, 33):
            layer_id = 33 - idx
            try:
                me.model = copy.deepcopy(orig_model)

                randomize_layer(me, idx)                
                res[f"Rnd_{layer_id}"] = self.lsc.explain(me, inp, catidx).cpu()
                for lidx in range(idx, 33):
                    randomize_layer(me, lidx)
                res[f"Csc_{layer_id}"] = self.lsc.explain(me, inp, catidx).cpu()
            finally:
                me.model = orig_model

        return res


