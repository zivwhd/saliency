import torch
import torchvision

import torch.nn as nn

import os, glob, json, pickle
import random, logging

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pdb

from saleval import *

def get_all_images():
    all_images = glob.glob("imagenet100/*/*.JPEG")
    return all_images

def get_selected_images(selection):
    selected_images = {}
    for x in selection:
        selected_images[x] = glob.glob(f"imagenet100/*/{x}.JPEG")[0]
    return selected_images

def image_name_to_class(info):    
    all_images = get_all_images()
    rv = {}
    #for ipath in all_images:
    #os.path.basename(x) : os.path.basename(os.path.dirname(x))



def show_image(path):
    img=Image.open(path)
    img=img.resize((224,224))    
    plt.title('input image', fontsize=18)
    plt.imshow(img)


class ModelEnv:

    def __init__(self, arch):
        self.arch = arch
        self.device = self.get_device()
        self.model = self.load_model(self.arch, self.device)
        self.shape = (224,224)

    def load_model(self, arch, dev):
        # Get a network pre-trained on ImageNet.
        model = torchvision.models.__dict__[arch](pretrained=True)
        # Switch to eval mode to make the visualization deterministic.
        model.eval()
        # We do not need grads for the parameters.
        for param in model.parameters():
            param.requires_grad_(False)

        model = model.to(dev)
        return model

    def narrow_model(self, catidx):
        return nn.Sequential(self.model, SelectKthLogit(catidx))
    
    def get_cam_target_layer(self):
        if self.arch == 'resnet50':
            return self.model.layer4
        raise Exception('Unexpected arch')
    
    def get_device(self, gpu=0):
        device = torch.device(
            f'cuda:{gpu}'
            if torch.cuda.is_available() and gpu is not None
            else 'cpu')
        return device

    def  get_image(self, path):
        img = Image.open(path)
        # Pre-process the image and convert into a tensor
        ## TODO: for which models are these transformation relevant
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.shape),
            torchvision.transforms.CenterCrop(self.shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
        ])

        x = transform(img).unsqueeze(0)
        return x.to(self.device)

    

### general utils

class ImageNetInfo:
    def __init__(self, path='dataset/imagenet_class_index.json'):
        with open(os.path.abspath(path), 'r') as read_file:
            class_idx = json.load(read_file)
            self.class_idx = class_idx
            self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            self.cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
            self.cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))} 



def get_result_path(variant, image_name, run=0, result_type="saliency"):
    return os.path.join("results", result_type, f"{variant}_{run}", image_name)

def get_saliency_path(variant, image_name, run=0):
    return get_result_path(variant=variant, image_name=image_name, run=run, result_type="saliency")

def save_saliency(obj, variant, image_name, run=0):    
    path = get_saliency_path(variant, image_name, run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def save_scores(scores_dict, image_name, run=0):
    for variant, scores in scores_dict.items():
        path = get_result_path(variant, image_name, run, result_type="scores")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as sf:
            pickle.dump(scores, sf)



class SelectKthSoftmax(nn.Module):
    def __init__(self, k):
        super(SelectKthSoftmax, self).__init__()
        self.k = k        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss  = nn.CrossEntropyLoss()

    def forward(self, x):        
        values =  self.softmax(x)
        result = values[...,self.k]
        
        return result
        
class SelectKthLogit(nn.Module):
    def __init__(self, k):
        super(SelectKthLogit, self).__init__()
        self.k = k        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss  = nn.CrossEntropyLoss()

    def forward(self, x):
        values = torch.stack([x], dim=-1)        
        #values = torch.stack((x, self.sigmoid(x), self.softmax(x), torch.logit(self.softmax(x))), dim=-1)
        result = values[...,self.k,:]
        return result
    

def create_saliency_data(me, algo, all_images, run_idx=0, exist_name=None, with_scores=False):

    info = ImageNetInfo()

    for itr, img in enumerate(all_images):    
            
        
        image_name = img.name
        image_path = img.path 

        pidx = image_name.find(".")
        if pidx > 0:
            image_name = image_name[0:pidx]

        if exist_name:
            progress_path = os.path.join("results", "progress", exist_name, image_name)
            if os.path.exists(progress_path):
                logging.info(f"## {itr} {image_path} {image_name} - Found skipping")
                continue

        inp = me.get_image(image_path)
        logits = me.model(inp).cpu()
        topidx = int(torch.argmax(logits))        
        logging.info(f"creating sal {itr} {image_path} {image_name} {topidx} {img.desc}")

        #mdl = nn.Sequential(me.model, SelectKthLogit(topidx))
        sal_dict = algo(me, inp, topidx)

        logging.info("done, saving")
        for variant, sal  in sal_dict.items():
            if variant.startswith("_"):
                continue
            save_saliency(sal, variant, image_name, run=run_idx)

        os.makedirs(os.path.dirname(progress_path), exist_ok=True)
        with open(progress_path, "wt") as pf:
            pf.write(".")
        
        if not with_scores:
            continue

        scores_dict = get_sal_scores(me, inp, sal_dict)
        save_scores(scores_dict, image_name, run=run_idx)

def get_sal_scores(me, inp, sal_dict):
    smodel = nn.Sequential(me.model, nn.Softmax(dim=1)) 
    scores_dict = {}
    flat_sal_dict = {}
    dinp = inp.cpu().detach()
    for key, data in sal_dict.items():        
        if key.startswith("_"):
            continue
        for idx in range(data.shape[1]):
            if idx > 0:
                continue
            saltns = data [idx:idx+1,...]
            sal_name = key #+ suffix_map[idx]
            
            sal = np.asarray(saltns[0].cpu())
            flat_sal_dict[sal_name] = saltns
            #print (inp.shape, sal.shape)
            del_metric = CausalMetric(smodel, "del",  me.shape[0], substrate_fn=torch.zeros_like)    
            #print(type(inp), type(sal))
            del_scores = del_metric.single_run(dinp, sal, verbose=0)

            ins_metric = CausalMetric(smodel, 'ins', me.shape[0], substrate_fn=blur)
            ins_scores = ins_metric.single_run(dinp, sal, verbose=0)

            scores_dict[sal_name] = scores = {}

            scores["del"] = del_scores
            scores["del_auc"] = auc(del_scores)
            scores["ins"] = ins_scores
            scores["ins_auc"] = auc(ins_scores)

            print("###", idx, sal_name, auc(del_scores), auc(ins_scores))
    return scores_dict

def show_sal_scores(img, scores_dict, sals_dict):
    plt.figure(figsize=(10, 3 * len(scores_dict)))

    idx = 1
    for name, scores in scores_dict.items():
                
        topidx = 0
        plt.subplot(len(scores_dict), 4, idx)
        isal = sals_dict[name][0].cpu()
        
        plt.title(name)
        plt.imshow(isal, cmap='coolwarm')#cmap='RdBu')
        plt.subplot(len(scores_dict), 4, idx+1)
        plt.imshow(img)    
        plt.imshow(isal, cmap='coolwarm', alpha=0.5)  # Set alpha for transparency
        #plt.axis('off')  # Hide the axis
        #plt.show()    
        plt.subplot(len(scores_dict), 4, idx+2)
        plot_scores("", "deletion", scores["del"])
        plt.subplot(len(scores_dict), 4, idx+3)
        plot_scores("", "insertion", scores["ins"])
        idx +=4
        
    plt.subplots_adjust(hspace=0.5)   
    plt.show()

def showsal(sal, img, caption="", quantile=0.9):
    stdsal = np.array( ((sal - sal.min()) / (sal.max()-sal.min())).unsqueeze(-1)) 
    stdsal = (stdsal > 0.7)
    
    plt.subplot(1, 3, 1)
    plt.title(caption)
    plt.imshow(sal, cmap='jet')#cmap='RdBu')
    plt.xticks([])  
    plt.yticks([])
    plt.subplot(1, 3, 2)
    plt.imshow(img)    
    plt.imshow(sal, cmap='jet', alpha=0.4)  # Set alpha for transparency
    plt.xticks([])  
    plt.yticks([])
    
    plt.subplot(1, 3, 3)
    bar = torch.quantile(sal, quantile)
    masked_img = ((sal > bar).unsqueeze(-1)).numpy() *img
    #img = img * 
    #plt.imshow((stdsal*img).astype(int))  # Set alpha for transparency
    plt.imshow(masked_img)
    plt.xticks([])  
    plt.yticks([])

    plt.show()    

def show_sal_dict(sals, img):
    for name, sal in sals.items():
        showsal(sal[0].cpu(), img, caption=name)


class CombSaliencyCreator:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, me, inp, catidx):
        res = {}
        for inner in self.inner:
            ires = inner(me, inp, catidx)
            res.update(ires)
        return res
        