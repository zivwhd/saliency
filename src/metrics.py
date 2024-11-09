import torch
import numpy as np
from sklearn.metrics import auc
from saliency.metrics import pic
from PIL import Image
import time

class Metrics:

    CONVENT_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    CONVNET_NORMALIZATION_STD = [0.229, 0.224, 0.225]

    def normalize(self, tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor

    
    def get_metrics(self, me, inp, img, saliency, info, nsteps=20, pred_only=True):

        model = me.model
        logits = model(inp).cpu()
        topidx = int(torch.argmax(logits))
        target = info.target

        ## del, pos
        pred_pos_auc, pred_del_auc = self.pert_metrics(model, inp, saliency[0], topidx, is_neg=False, nsteps=nsteps)
        pred_neg_auc, pred_ins_auc = self.pert_metrics(model, inp, saliency[0], topidx, is_neg=True, nsteps=nsteps)
        pred_adp, pred_pic = self.get_adp_pic(model, inp, saliency[0], topidx)
        pred_sic = self.get_sic(me, inp, img, saliency, target)
        pred_aic = self.get_aic(me, inp, img, saliency, target)

        if pred_only:
            return dict(
                pred_pos_auc=pred_pos_auc,
                pred_neg_auc=pred_neg_auc,
                pred_del_auc=pred_del_auc,
                pred_ins_auc=pred_ins_auc,
                pred_adp=pred_adp,
                pred_pic=pred_pic,
                pred_aic=pred_aic,
                pred_sic=pred_sic,
                )

        assert False, "missing aic,sic impl"
        if target == topidx:
            target_pos_auc, target_del_auc = pred_pos_auc, pred_del_auc
            target_neg_auc, target_ins_auc = pred_neg_auc, pred_ins_auc 
            target_adp, target_pic = pred_adp, pred_pic, pred_aic, pred_sic
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
            pred_aic=0,
            pred_sic=0,

            target_pos_auc=target_pos_auc,
            target_neg_auc=target_neg_auc,
            target_del_auc=target_del_auc,
            target_ins_auc=target_ins_auc,

            target_adp=target_adp,
            target_pic=target_pic,
            target_aic=target_aic,
            target_sic=target_sic,

        )


    def get_sic(self, me, inp, img, saliency, target):
        device = inp.device
        transform = me.get_transform()
        model = me.model
        
        def predict(image_batch):             
            nonlocal transform, model, device
            
            inp = torch.stack([transform(Image.fromarray(x)) for x in image_batch]).to(device)
            logits = model(inp)
            probs = torch.softmax(logits, 1)
            score = probs[:, target].detach().cpu()
            return score.numpy()
        
        return self.get_pic_auc(model, inp, img, saliency, target, predict)

    def get_aic(self, me, inp, img, saliency, target):        
        device = inp.device
        transform = me.get_transform()
        model = me.model
        
        def predict(image_batch):            
            nonlocal transform, model, device                        
            inp = torch.stack([transform(Image.fromarray(x)) for x in image_batch]).to(device)
            logits = model(inp)            
            image_class = torch.argmax(logits, dim=1)        
            score = (image_class.detach().cpu() == target).float()
            return score.numpy()
        
        return self.get_pic_auc(model, inp, img, saliency, target, predict)


    def get_pic_auc(self, model, inp, img, saliency, target,
                pred_func,
                num_data_points = 20, fraction=0.01):
        random_mask = pic.generate_random_mask(image_height=inp.shape[-1], image_width=inp.shape[-2], fraction=fraction)
        
        saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]
        nimg = np.array(img)
        start = time.time()
        try:
            metric = pic.compute_pic_metric(
                img=nimg,
                saliency_map=saliency[0].numpy(),
                random_mask=random_mask,
                pred_func=pred_func,
                min_pred_value=0.5,
                saliency_thresholds=saliency_thresholds,
                keep_monotonous=True,
                num_data_points=num_data_points)            
            return metric.auc * 100.0
    
        except pic.ComputePicMetricError as e:        
            return -10000.0


    def get_adp_pic(self, model, inp, saliency, target):
        logits = model(inp).cpu()
        probs = torch.softmax(logits, dim=1)

        nsal = saliency
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
        accuracy = (target == probs[0].argmax()).float()


        adp = (torch.maximum(x - y, torch.zeros_like(x)) / x).mean() * 100
        pic = torch.where(x < y, 1.0, 0.0).mean() * 100

        return float(adp), float(pic)
        

    def pert_metrics(self, model, inp, saliency, target, is_neg=False, nsteps=100, 
                     with_steps=False, finish=None):
        org_shape = inp.shape
        data = inp.clone().cpu()

        vis = saliency

        if is_neg:
            vis = -vis

        if finish is None:
            finish = 0
        else:
            finish = finish.reshape(org_shape[0], org_shape[1], -1)

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
                _data = _data.scatter_(-1, idx.to(_data.device), 0)
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
        #print(accuracy_auc, prob_auc)
        if with_steps:
            return (accuracy_auc, prob_auc, accuracy, probs)
        return (accuracy_auc, prob_auc)