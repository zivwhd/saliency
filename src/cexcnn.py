import torch
import cv2
import numpy as np
import logging

class CexCnnSaliencyCreator:
    def __init__(self, quanttiles=[0.25,0.5,0.75,0.9, 0.95]):
        self.quantiles = quanttiles

    def __call__(self, me, inp, catidx):
        llayer = me.get_cex_conv_layer()

        resp = {}
        orig_probs = torch.softmax(me.model(inp)[0], dim=0)
        delta_prob_list = []
        with torch.no_grad():
            logging.debug("CexCnn: iterating over filters")
            for idx in range(llayer.out_channels):
                keep = llayer.weight.data[idx].clone()
                llayer.weight.data[idx] = 0.0
                cf_probs = torch.softmax(me.model(inp)[0], dim=0)
                #print(idx, cf_probs[catidx], cf_probs[catidx]-orig_probs[catidx])
                delta = cf_probs[catidx]-orig_probs[catidx]
                delta_prob_list.append(delta.detach().cpu())
                llayer.weight.data[idx] = keep
                keep = None
                torch.cuda.ipc_collect()
            logging.debug("done with filters")
            delta_prob = torch.stack(delta_prob_list)
            threshold = 0             
            for qntl in self.quantiles:
                resp[qntl] = ((delta_prob > threshold) & (delta_prob > delta_prob.quantile(qntl)))

        pooled_grads = None
        activations = None

        def save_grads(grad):
            global pooled_grads
            poolded_grads = grad

        def fhook(module, input, output):
            nonlocal activations 
            activations = output
            print("set activations")
            #print(type(module), type(input), type(output))
            output.register_hook(save_grads)
                
        hook = llayer.register_forward_hook(fhook)

        try: 
            output = me.model(inp)
        finally:
            hook.remove()
        loss = output[0, catidx]
        me.model.zero_grad()
        loss.backward()

        sal_dict = {}
        for qntl, qresp in resp.items():
            heatmap = (activations.cpu().squeeze(0) * qresp.unsqueeze(1).unsqueeze(1)).mean(dim=0).detach().clone()
            sal = torch.tensor(cv2.resize(heatmap.numpy(), (224,224))).unsqueeze(0)
            sal_dict[f"CexCnn_{qntl}"] = sal
        return sal_dict