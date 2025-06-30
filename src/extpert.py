from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.utils import get_device
import logging, time, sys
import torch


def qmet(smdl, inp, sal, steps):
    with torch.no_grad():    
        bars = sal.quantile(steps).unsqueeze(1).unsqueeze(1)
        del_masks = (sal.unsqueeze(0) < bars)
        ins_masks = (sal.unsqueeze(0) > bars)        
        del_pred = smdl(del_masks.unsqueeze(1) * inp)
        ins_pred = smdl(ins_masks.unsqueeze(1) * inp)
        del_auc = ((del_pred[1:]+del_pred[0:-1])*0.5).mean() 
        ins_auc = ((ins_pred[1:]+ins_pred[0:-1])*0.5).mean() 
        return del_auc.cpu().tolist(), ins_auc.cpu().tolist()

class ExtPertSaliencyCreator:

    def __init__(self, single=True):
        self.single=single

    def __call__(self, me, inp, catidx):   
        start_time = time.time()
        smdl = me.narrow_model(catidx, with_softmax=True)
        threshold = smdl(inp)[0,0] * 0.6
        
        desc = f"ExtPert"        
        areas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

        #print("Category:", catidx, file=sys.stderr)
        masks, _ = extremal_perturbation(
            me.model, inp, catidx,
            reward_func=contrastive_reward,
            debug=False,
            areas=areas)

        logging.info(f"got masks {masks.shape} [{inp.shape} {threshold}]")

        res = {}
        selected = None
        met_selected = None        
        metric_steps = torch.tensor([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]).to(inp.device)
        for idx,mask in enumerate(masks):
            desc = f'ExtPert_{areas[idx]}'


            cmask = mask.detach().clone().cpu() ## 1,224,224
            
            res[desc] = cmask
            prob = me.model(inp * mask.unsqueeze(0))[0,0]

            if (selected is None) and ((prob >= threshold) or (idx == len(masks) -1)):
                selected = cmask
                res[f'ExtPertS'] = selected

            sdel, sins = qmet(smdl, inp, mask.squeeze(0), steps=metric_steps )
            mscore = sins-sdel

            if met_selected is None or mscore > met_selected[0]:
                met_selected = (mscore, cmask)
                res[f'ExtPertM'] = cmask
        duration = time.time() - start_time
        logging.info(f"run-time: {duration}")
        if self.single:
            res = {'ExtPertM' : res[f'ExtPertM']}
        return res

