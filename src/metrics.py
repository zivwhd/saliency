import torch
import numpy as np
from sklearn.metrics import auc

class Metrics:

    CONVENT_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    CONVNET_NORMALIZATION_STD = [0.229, 0.224, 0.225]

    def normalize(self, tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor

    
    def get_metrics(self, model, inp, saliency, info, nsteps=100):

        logits = model(inp).cpu()
        topidx = int(torch.argmax(logits))
        target = info.target

        ## del, pos
        pred_pos_auc, pred_del_auc = self.pert_metrics(model, inp, saliency[0], topidx, is_neg=False, nsteps=nsteps)
        pred_neg_auc, pred_ins_auc = self.pert_metrics(model, inp, saliency[0], topidx, is_neg=True, nsteps=nsteps)
        pred_adp, pred_pic = self.get_adp_pic(model, inp, saliency[0], topidx)

        if target == topidx:
            target_pos_auc, target_del_auc = pred_pos_auc, pred_del_auc
            target_neg_auc, target_ins_auc = pred_neg_auc, pred_ins_auc 
            target_adp, target_pic = pred_adp, pred_pic
        else:
            target_pos_auc, target_del_auc = self.pert_metrics(model, inp, saliency[1], target, is_neg=False, nsteps=nsteps)
            target_neg_auc, target_ins_auc = self.pert_metrics(model, inp, saliency[1], target, is_neg=True, nsteps=nsteps)
            target_adp, target_pic = self.get_adp_pic(model, inp, saliency[1], target)

        

        return dict(
            pred_pos_auc=pred_pos_auc,
            pred_neg_auc=pred_neg_auc,
            pred_del_auc=pred_del_auc,
            pred_ins_auc=pred_ins_auc,
            pred_adp=pred_adp,
            pred_pic=pred_pic,

            target_pos_auc=target_pos_auc,
            target_neg_auc=target_neg_auc,
            target_del_auc=target_del_auc,
            target_ins_auc=target_ins_auc,

            target_adp=target_adp,
            target_pic=target_pic,
        )



    def get_adp_pic(self, model, inp, saliency, target):
        logits = model(inp).cpu()
        probs = torch.softmax(logits, dim=1)

        nsal = torch.maximum(saliency, torch.zeros(saliency.shape))

        salmax, salmin = nsal.max(), nsal.min()
        if salmax > salmin:
            mask = (nsal - salmin) / (salmax-salmin)
        else: 
            mask = torch.zeros(nsal.shape)

        masked_inp = inp * mask.to(inp.device).unsqueeze(0)
        pred_mask = model(masked_inp)
        probs_mask = torch.softmax(pred_mask, dim=1)

        x = probs[0,target]
        y = probs_mask[0, target]

        adp = (torch.maximum(x - y, torch.zeros_like(x)) / x).mean() * 100
        pic = torch.where(x < y, 1.0, 0.0).mean() * 100

        return float(adp), float(pic)
        

    def pert_metrics(self, model, inp, saliency, target, is_neg=False, nsteps=100):
        org_shape = inp.shape
        data = inp.clone().cpu()

        vis = saliency

        if is_neg:
            vis = -vis

        vis = vis.reshape(org_shape[0], -1)

        accuracy = []
        probs = []

        perturbation_steps =  torch.arange(nsteps+1, dtype=torch.float32)  / nsteps
        
        for idx, part in enumerate(perturbation_steps):
            _data = data.clone()        
            perturbation_size = int((vis.numel() * part).tolist())
            #print("### pert_size", vis.numel(), part, pertubation_size)
            if perturbation_size:
                #print("###", vis.numel() , vis.device)
                _, idx = torch.topk(vis, perturbation_size, dim=-1)  # get top k pixels        
                idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
                _data = _data.reshape(org_shape[0], org_shape[1], -1)
                _data = _data.scatter_(-1, idx, 0)
                _data = _data.reshape(*org_shape)

            norm_data = _data
            #_norm_data = self.normalize(
            #    tensor=_data,
            #    mean=self.CONVENT_NORMALIZATION_MEAN,   
            #    std=self.CONVNET_NORMALIZATION_STD)
            
            out = model(norm_data.to(inp.device))
            #print(out.shape)
            target_class = out.max(1, keepdim=True)[1].squeeze(1)
            correct = float((target == target_class))  #.type(target.type()).data.cpu().numpy()
            #print("### targets", perturbation_size, target, target_class)
            #print("### correct", correct)
            #num_correct_pertub[i, perturb_index:perturb_index + len(temp)] = temp
            probs_pertub = torch.softmax(out, dim=1)

            accuracy.append(correct)
            #print(probs_pertub.shape)
            probs.append(float(probs_pertub[0,target]))
        
        accuracy_auc  = auc(perturbation_steps, accuracy) * 100
        prob_auc = auc(perturbation_steps, probs) * 100
        print(accuracy_auc, prob_auc)
        return (accuracy_auc, prob_auc)