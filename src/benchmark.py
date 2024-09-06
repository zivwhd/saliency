import torch
import torchvision

import torch.nn as nn
import pandas as pd

import os, glob, json, pickle
import random, logging

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pdb
from collections import defaultdict

from saleval import *

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



def get_result_path(variant, image_name, run=0, result_type="saliency"):
    return os.path.join("results", result_type, f"{variant}_{run}", image_name)

def get_all_results(subset=None):
    all_sals = glob.glob(os.path.join("results", "saliency", "*", "*"))
    if subset:
        all_sals = [x for x in all_sals if os.path.basename(x) in subset]
    return all_sals


def get_saliency_path(variant, image_name, run=0):
    return get_result_path(variant=variant, image_name=image_name, run=run, result_type="saliency")

def save_saliency(obj, variant, image_name, run=0):    
    path = get_saliency_path(variant, image_name, run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def save_scores(scores_dict, image_name, run=0, update=False):
    for variant, scores in scores_dict.items():
        path = get_result_path(variant, image_name, run, result_type="scores")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if update and os.path.exists(path):
            with open(path, "rb") as sof:
                orig_scores = pickle.load(sof)
                uscores = {}
                uscores.update(orig_scores)
                uscores.update(scores)
                scores = uscores

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
    

def acquire_file(destination_path):
    try:
        # Create a temporary file in the same directory as the destination path
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(destination_path), suffix='.tmp', mode='w', encoding='utf-8')
        temp_file_name = temp_file.name
        temp_file.close()  # Close the temporary file so we can rename it

        # Attempt to rename the temporary file to the destination path
        os.rename(temp_file_name, destination_path)

        # Rename successful, return True
        return True
    except OSError as e:
        # If renaming fails (e.g., file already exists), remove the temporary file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)
        logging.info(f"Failed to acquire file: {e}")
        return False
    
def create_saliency_data(me, algo, all_images, run_idx=0, exist_name=None, with_scores=False):

    for itr, img in enumerate(all_images):    
        
        image_name = img.name
        image_path = img.path 

        pidx = image_name.find(".")
        if pidx > 0:
            image_name = image_name[0:pidx]

        if exist_name:
            progress_path = os.path.join("results", "progress", f"{exist_name}_{image_name}")            
    
            if os.path.exists(progress_path):
                logging.info(f"## {itr} {image_path} {image_name} - Found skipping")
                continue

            acquire_path = os.path.join("results", "acquire", f"{exist_name}_{image_name}")
            if not acquire_file(acquire_path):
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

        scores_dict = get_sal_scores(me, inp, sal_dict, with_breakdown=False)
        save_scores(scores_dict, image_name, run=run_idx, update=True)

def get_sal_scores(me, inp, sal_dict, with_breakdown=True):
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

            if with_breakdown:
                scores["del"] = del_scores
                scores["ins"] = ins_scores
            scores["ins_auc"] = auc(ins_scores)
            scores["del_auc"] = auc(del_scores)

            logging.debug(f"scores  {idx}, {sal_name}, {scores['del_auc']}, {scores['ins_auc']}")
    return scores_dict

def create_scores(me, images, result_paths, update=True):
    for path in result_paths:
        
        image_name = os.path.basename(path)
        variant = os.path.basename(os.path.dirname(path))
        logging.debug(f"checking: {path} name={image_name} variant={variant}")
        if image_name not in images:
            continue

        logging.debug(f"handling {path}")
        info = images[image_name]

        inp = me.get_image(info.path)
        sal_dict = {variant : torch.load(path)}
        scores_dict = get_sal_scores(me, inp, sal_dict)
        save_scores(scores_dict, image_name, update=update)



def load_scores_df(variant_names=None, base_path="results/scores"):

    if variant_names is None:
        variant_names = [os.path.basename(x) for x in glob.glob(os.path.join(base_path, "*"))]

    def append_row(res, **kwargs):
        for key, value in kwargs.items():
            res[key].append(value)
        
    res = defaultdict(list)
    for variant in variant_names:
        score_files = glob.glob(os.path.join(base_path, variant, "*"))
        for scores_path in score_files:
            image_name = os.path.basename(scores_path)
            with open(scores_path, "rb") as sf:
                scores = pickle.load(sf)
            
            append_row(
                res, 
                image=image_name, variant=variant, del_auc=scores["del_auc"], ins_auc=scores["ins_auc"])
    return pd.DataFrame(res)

def summarize_scores_df(df):
    smry =df.groupby('variant').agg(
        mean_del_auc=('del_auc', 'mean'),
        mean_ins_auc=('ins_auc', 'mean'),
        row_count=('variant', 'size')
    ).reset_index()
    return smry    


class CombSaliencyCreator:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, me, inp, catidx):
        res = {}
        for inner in self.inner:
            ires = inner(me, inp, catidx)
            res.update(ires)
        return res
        