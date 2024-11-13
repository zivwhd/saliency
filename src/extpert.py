from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.utils import get_device
import logging

class ExtPertSaliencyCreator:

    def __init__(self):
        pass

    def __call__(self, me, inp, catidx):   
        desc = f"ExtPert"        
        areas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
        masks, _ = extremal_perturbation(
            me.model, inp, catidx,
            reward_func=contrastive_reward,
            debug=False,
            areas=areas)

        res = {}
        for idx, area in enumerate(areas):
            res[f'ExtPert_{area}'] = masks[idx:idx+1]
    
        return res