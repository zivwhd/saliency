from lcpe import *
import copy

import torch
from timm import create_model
import logging
import torch
import torchvision


def set_seed(seed=42):
    seed = 42
    torch.manual_seed(seed)
    # If using CUDA:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-G

PMODEL = None
VMODEL = None
def randomize_layer_(me, layer_index):
    global PMODEL
    assert me.arch == "resnet50"
    if PMODEL is None:
        PMODEL = torchvision.models.resnet50(pretrained=False)
    
    modified = False
    conv_layers = [x for x in me.model.named_parameters() if "conv" in x[0]]
    rnd_conv_layers = [x for x in PMODEL.named_parameters() if "conv" in x[0]]
    param = conv_layers[layer_index][1]
    param.data = rnd_conv_layers[layer_index][1].data.to(me.device) ##torch.randn_like(param.data) 


def randomize_layer(me, start_idx, end_idx=None):
    if end_idx is None:
        end_idx = start_idx

    global PMODEL
    global VMODEL
    if me.arch == "resnet50":
        if PMODEL is None:
            PMODEL = torchvision.models.resnet50(pretrained=False)

        state_dict = me.model.state_dict()
        other_state_dict = PMODEL.state_dict()
        import re
        lidx = 0
        selected = []

        ptrn = re.compile('(conv[0-9]|downsample|fc).*weight')
        layer_keys = [key for key in state_dict.keys() if ptrn.search(key)]
        total = len(layer_keys)
        select = []
        for key in list(state_dict.keys()):
            if lidx >= total - start_idx and lidx <= total - end_idx:
                select.append(key)            
            lidx += (key in layer_keys)
        
        for key in select:
            state_dict[key] = other_state_dict[key]
        me.model.load_state_dict(state_dict)
    if me.arch == "vgg16":
        if VMODEL is None:
            VMODEL = torchvision.models.vgg16(pretrained=False)
        state_dict = me.model.state_dict()
        other_state_dict = VMODEL.state_dict()
        import re

        from collections import defaultdict

        layer_key_list = []
        ptrn = re.compile('(features.[0-9]+|classifier.[0-9]+).(weight|bias)')
        for key in list(state_dict.keys()):
            match = ptrn.match(key)
            if not match:
                continue
            layer_key = match.group(1)
            if layer_key not in layer_key_list:
                layer_key_list.append(layer_key)
        layer_key_list.reverse()

        for key in list(state_dict.keys()):
            match = ptrn.match(key)
            if not match:
                continue
            layer_key = match.group(1)
            if layer_key not in layer_key_list:
                continue
            layer_idx = layer_key_list.index(layer_key) + 1            
            if layer_idx >= end_idx and layer_idx <= start_idx:
                state_dict[key] = other_state_dict[key]
                logging.info(f"updating {key}")

        me.model.load_state_dict(state_dict)        
    else:
        assert False, f"unexpected arch {me.arch}"


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
        
        #conv_layers = [x for x in me.model.named_parameters() if "conv" in x[0]]
        #nlayers = len(conv_layers)
        nlayers = 16
        res["Base"] = self.lsc.explain(me, inp, catidx).cpu()

        for idx in range(1,nlayers+1):
            layer_id = idx
            try:
                me.model = copy.deepcopy(orig_model)
                
                randomize_layer(me, idx, idx)
                res[f"Rnd_{layer_id}"] = self.lsca.explain(me, inp, catidx).cpu()
                randomize_layer(me, idx, 0)
                res[f"Csc_{layer_id}"] = self.lsca.explain(me, inp, catidx).cpu()
            finally:
                me.model = orig_model

        return res


