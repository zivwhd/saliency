from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.utils import get_device
import logging

class ExtPertSaliencyCreator:

    def __init__(self):
        pass

    def __call__(self, me, inp, catidx):   
        smdl = me.narrow_model(catidx, with_softmax=True)
        threshold = smdl(inp)
        
        desc = f"ExtPert"        
        areas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

        masks, _ = extremal_perturbation(
            me.model, inp, catidx,
            reward_func=contrastive_reward,
            debug=False,
            areas=areas)

        logging.info(f"got masks {masks.shape} [{inp.shape}]")

        for mask in masks:
            prob = me.model(inp * mask)
            if prob >= threshold:
                return { 'ExtPert' : mask.detach().clone().cpu() }
            
        return { 'ExtPert' : masks[-1].detach().clone().cpu() }

