from lcpe import *
import copy

import torch
from timm import create_model

def replace_layer_weights_with_random(model, layer_index):
    """
    Replaces the weights of a layer in the model at the specified index with random values.
    
    Args:
        model: PyTorch model (e.g., DenseNet201 from timm).
        layer_index: Index of the layer in model.named_parameters().
    """
    # List all named parameters in the model
    named_params = list(model.named_parameters())
    
    if layer_index < 0 or layer_index >= len(named_params):
        raise IndexError(f"Layer index {layer_index} is out of bounds. Model has {len(named_params)} layers.")
    
    # Get the layer name and current weights
    layer_name, param = named_params[layer_index]
    print(f"Replacing weights for layer '{layer_name}' with random values.")
    
    # Generate random weights with the same shape
    random_weights = torch.randn_like(param.data)
    
    # Replace the weights
    param.data = random_weights
    
    print(f"Layer '{layer_name}' updated with random weights.")

# Example Usage
# Load DenseNet201 model
model = create_model("densenet201", pretrained=True)

# Replace weights for the 10th layer (example)
replace_layer_weights_with_random(model, layer_index=10)

# Verify the weights
named_params = list(model.named_parameters())
print(f"Updated weights for layer '{named_params[10][0]}':")
print(named_params[10][1].data)



def randomize_layer(me, layer_index):    
    modified = False
    for name, param in me.model.named_parameters():
        if (f".denselayer{layer_index}." not in name):
            continue
        modified = True
        logging.info("randomizing layer weights {name}")
        random_weights = torch.randn_like(param.data)        
        # Replace the weights
        param.data = random_weights
    assert modified

class SanityCreator:
    def __init__(self):
        self.lsc = CompExpCreator(nmasks=500, segsize=40, c_mask_completeness=1.0, c_magnitude=0.01, 
                                  c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=None)

    def __call__(self, me, inp, catidx, data=None):

        orig_model = me.model
        res = {}
        
        res["Base"] = self.lsc.explain(me, inp, catidx)

        for idx in range (1, 33):
            layer_id = 33 - idx
            try:
                me.model = copy.deepcopy(me.model)

                randomize_layer(me, idx)                
                res[f"Rnd_{layer_id}"] = self.lsc.explain(me, inp, catidx)
                for lidx in range(idx, 33):
                    randomize_layer(me, lidx)
                res[f"Csc_{layer_id}"] = self.lsc.explain(me, inp, catidx)
            finally:
                me.model = orig_model

        return res
