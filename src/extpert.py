from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.utils import get_device
import logging

class ExtPertSaliencyCreator:

    def __init__(self):
        pass

    def __call__(self, me, inp, catidx):   
        smdl = me.narrow_model(catidx, with_softmax=True)
        threshold = smdl(inp)[0,0]
        
        desc = f"ExtPert"        
        areas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

        masks, _ = extremal_perturbation(
            me.model, inp, catidx,
            reward_func=contrastive_reward,
            debug=False,
            areas=areas)

        logging.info(f"got masks {masks.shape} [{inp.shape} {threshold}]")

        res = {}
        selected = None
        for idx,mask in enumerate(masks):
            desc = f'ExtPert_{areas[idx]}'
            cmask = mask.detach().clone().cpu()
            res[desc] = cmask
            prob = me.model(inp * mask.unsqueeze(0))[0,0]
            
            if (selected is None) and ((prob >= threshold) or (idx == len(masks) -1)):
                selected = cmask
                res[f'ExtPertS'] = selected
            
        return res

