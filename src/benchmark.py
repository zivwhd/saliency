import torch
import torchvision

import torch.nn as nn
import pandas as pd

import os, glob, json, pickle
import random, logging, socket

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pdb
from collections import defaultdict

import timm, torchray
import torchray.benchmark

from saleval import *
from metrics import *
from pklread import read_pickle_files

class ModelEnv:

    def __init__(self, arch):
        self.arch = arch
        self.device = self.get_device()
        self.model = self.load_model(self.arch, self.device)
        self.shape = (224,224)

    def load_model(self, arch, dev):
        if arch.startswith('voc_'):
            model_arch = arch.replace('voc_','')
            model = torchray.benchmark.models.get_model(arch=model_arch, dataset="voc", convert_to_fully_convolutional=False)

        elif arch == 'vgg16NT':
            model = torchvision.models.vgg16(pretrained=False)
        elif arch == 'vgg16RT':            
            model = torchvision.models.vgg16(pretrained=False)
            output_weights_path = 'models/vgg16_retrained_n.pth'
            model.load_state_dict(torch.load(output_weights_path))

        elif 'resnet' in arch or 'vgg' in arch:
        # Get a network pre-trained on ImageNet.
            model = torchvision.models.__dict__[arch](pretrained=True)
            #for param in model.parameters():
            #    param.requires_grad_(False)        
        elif 'vit' in arch or 'convnext' in arch or 'densenet' in arch:
            model = timm.create_model(arch, pretrained=True)
        else:
            assert False, "unexpected arch"
        
        model.eval()        
        model = model.to(dev)
        return model

    def narrow_model(self, catidx, with_softmax=False):
        if "voc" in self.arch:
            modules = (
                [self.model] + 
                #([nn.Sigmoid()] if with_softmax else []) +
                [SelectKthLogit(catidx)])
        else:
            modules = (
                [self.model] + 
                ([nn.Softmax(dim=1)] if with_softmax else []) +
                [SelectKthLogit(catidx)])

        return nn.Sequential(*modules)
        
    def get_cam_target_layer(self):
        if self.arch == 'resnet50':
            return self.model.layer4[-1]
            #return self.model.layer4
        
        elif self.arch == 'vgg16':
            return self.model.features[-1]

        elif self.arch == 'densenet201':
            return self.model.features[-1]

        elif self.arch == 'convnext_base':
            return self.model.stages[-1].blocks[-1]
                        
        raise Exception('Unexpected arch')
    
    def get_cex_conv_layer(self):
        
        if self.arch == 'resnet50':
            return self.model.layer4[-1].conv3

        elif self.arch == 'vgg16':
            return self.model.features[-3]

        elif self.arch == 'convnext_base':
            return self.model.stages[-1].blocks[-1].conv_dw

        raise Exception('Unexpected arch')

    def get_device(self, gpu=0):
        device = torch.device(
            f'cuda:{gpu}'
            if torch.cuda.is_available() and gpu is not None
            else 'cpu')
        return device

    def get_transform(self):    
        if "voc" in self.arch:
            #print("voc transform 3")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to (224, 224)
                transforms.ToTensor(),  # Convert to tensor (scales to [0,1])
                transforms.Lambda(lambda x: x * 255.0)  # Multiply by 255
            ])
        elif 'resnet' in self.arch or 'vgg' in self.arch or 'convnext' in self.arch or 'densenet' in self.arch:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.shape),
                torchvision.transforms.CenterCrop(self.shape),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
            ])
        elif 'vit' in self.arch:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            assert False, "unexpected arch"
        return transform

    def get_image_ext(self, path):
        img = Image.open(path)
        # Pre-process the image and convert into a tensor
        ## TODO: for which models are these transformation relevant        
        transform = self.get_transform()
        x = transform(img).unsqueeze(0)
        return img, x.to(self.device)

    def get_image(self, path):
        return self.get_image_ext(path)[1]
    

### general utils



def get_result_path(model_name, variant, image_name, run=0, result_type="saliency"):    
    return os.path.join("results", model_name, result_type, f"{variant}_{run}", image_name)

def get_all_results(model_name, subset=None): ## PPAA
    all_sals = glob.glob(os.path.join("results", model_name, "saliency", "*", "*"))
    if subset:
        all_sals = [x for x in all_sals if os.path.basename(x) in subset]
    return all_sals


def get_saliency_path(model_name, variant, image_name, run=0):
    return get_result_path(model_name=model_name, variant=variant, image_name=image_name, run=run, result_type="saliency")

def save_saliency(obj, model_name, variant, image_name, run=0):     ## PPAA
    path = get_saliency_path(model_name, variant, image_name, run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)

def save_scores(scores_dict, model_name, image_name, run=0, update=False, extended=False, equant=False):
    for variant, scores in scores_dict.items():
        result_type = get_ext_mark(extended=extended, equant=equant) + "scores"

        path = get_result_path(model_name,variant, image_name, run, result_type=result_type)
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
        if type(self.k) == int:            
            values = torch.stack([x], dim=-1)
            result = values[:,self.k,:]            
        else:
            result = x[:, self.k]                            
        
        return result
    

    
def create_saliency_data(me, algo, all_images, run_idx=0, with_scores=False, skip_img_error=True,
                         pred_only=True):

    for itr, info in enumerate(all_images):    
        
        image_name = info.name
        image_path = info.path 
        target = info.target

        pidx = image_name.find(".")
        if pidx > 0:
            image_name = image_name[0:pidx]

        try:
            img, inp = me.get_image_ext(image_path)
        except:
            logging.exception("Failed getting image")
            if skip_img_error:
                logging.info("Skipping")
                continue

        logits = me.model(inp).cpu()
        topidx = int(torch.argmax(logits))        
        logging.info(f"creating sal {itr} {image_path} {image_name} {topidx} {info.desc}")

        
        multi_target = getattr(algo, "multi_target", False)

        if pred_only:
            sal_dict = algo(me, inp, topidx)
        elif target == topidx:
            top_sal_dict = algo(me, inp, topidx)        
            sal_dict = {key : torch.concat([x,x], dim=0) for key, x in top_sal_dict.items()}
        elif multi_target:
            logging.debug("multi_target")
            sal_dict = algo(me, inp, (topidx, target) )
            for msal in sal_dict.values():
                assert (msal.shape[0] == 2)            
        else:
            top_sal_dict = algo(me, inp, topidx)
            target_sal_dict = algo(me, inp, target)
            assert (target_sal_dict.keys() == top_sal_dict.keys())
            sal_dict = {key : torch.concat([top_sal_dict[key], target_sal_dict[key]], dim=0) for key in top_sal_dict.keys()}


        logging.info("done, saving")
        for variant, sal  in sal_dict.items():
            if variant.startswith("_"):
                continue
            save_saliency(sal, me.arch, variant, image_name, run=run_idx)
        
        if not with_scores:
            continue

        scores_dict = get_sal_scores(me, inp, img, info, sal_dict)
        save_scores(scores_dict, me.arch, image_name, run=run_idx, update=True)

def get_sal_scores(me, inp, img, info, sal_dict, extended=False, equant=False):
    metrics = Metrics()    
    if equant:
        return {
            name: metrics.get_quantus_metrics(me, inp, img, sal, info)            
            for name, sal in sal_dict.items()
        }    

    if extended:
        return {
            name : metrics.get_ext_metrics(me, inp, img, sal, info)
            for name, sal in sal_dict.items()
        }
    else:
        return {
            name : metrics.get_metrics(me, inp, img, sal, info)
            for name, sal in sal_dict.items()
        }
    
    
def get_sal_scores_(me, inp, info, sal_dict, with_breakdown=True):
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

def get_score_name(path):
    return (os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path))

def create_scores(me, result_paths, images, update=True, extended=False, equant=False):
    for path in result_paths:
        
        image_name = os.path.basename(path)
        variant = os.path.basename(os.path.dirname(path))
        logging.debug(f"checking: {path} name={image_name} variant={variant}")

        #if image_name not in images:
        #    continue

        logging.debug(f"handling {path}")
        info = images[image_name]

        img, inp = me.get_image_ext(info.path)
        sal_dict = {variant : torch.load(path).float()}
        scores_dict = get_sal_scores(me, inp, img, info, sal_dict, extended=extended, equant=equant)
        save_scores(scores_dict, me.arch, image_name, update=update, extended=extended, equant=equant)


def get_ext_mark(extended=False, equant=False):
    if equant:
        return "q"
    elif extended:
        return "e"
    else:
        return ""

def load_scores_df(model_name, variant_names=None, base_path=None, filter_func=None, dist=True, extended=False, equant=False):
    if base_path is None:
        emark = get_ext_mark(extended=extended, equant=equant)        
        base_path = os.path.join("results", model_name, f"{emark}scores")

    if variant_names is None:
        variant_names = [os.path.basename(x) for x in glob.glob(os.path.join(base_path, "*"))]

    variant_names = [x for x in variant_names if filter_func(x)]    

    def append_row(res, **kwargs):
        for key, value in kwargs.items():
            if "target_" not in key:
                res[key].append(value)
        
    res = defaultdict(list)
    for variant in variant_names:        
        score_files = glob.glob(os.path.join(base_path, variant, "*"))

        
        if not dist:
            scores_list = []
            for scores_path in score_files:
                image_name = os.path.basename(scores_path)
                logging.debug(f"loading scors: {scores_path}")
                with open(scores_path, "rb") as sf:
                    scores = pickle.load(sf)
                append_row(
                    res, 
                    model=model_name, image=image_name, variant=variant, 
                    **scores)    

        else:
            scores_list = read_pickle_files(score_files)
            for scores_path, scores in scores_list:
                image_name = os.path.basename(scores_path)
                append_row(
                    res, 
                    model=model_name, image=image_name, variant=variant, 
                    **scores)    
            
    return pd.DataFrame(res)


def bootstrap_ci(series, B=1000, alpha_low=5, alpha_high=95):
    data = series.to_numpy()
    N = len(data)
    if N == 0:
        return pd.Series([np.nan, np.nan])  # handle empty groups

    means = np.array([
        np.mean(np.random.choice(data, size=N, replace=True))
        for _ in range(B)
    ])
    lower = np.percentile(means, alpha_low)
    upper = np.percentile(means, alpha_high)
    return pd.Series([lower, upper])

def summarize_scores_df(df, extended=False, equant=False):
    meta_cols = ["model","image","variant"]
    confidence_cols = ["pred_ins","pred_del"]
    metric_cols = [x for x in df.columns if x not in meta_cols]
    confidence_cols = [x for x in metric_cols if x in confidence_cols]

    if extended:
        metrics = {f"mean_{x}" : (x, lambda x: x[x >= 0].mean()) for x in metric_cols}
        metrics.update({f"count_{x}" : (x, lambda x: (x >=0).sum()) for x in metric_cols})
    else:
        metrics = {f"mean_{x}" : (x, 'mean') for x in metric_cols}
        for mcol in confidence_cols:
            metrics.update({
                f"ci05_{mcol}" : (mcol, lambda x:  bootstrap_ci(x)[0]),
                f"ci95_{mcol}" : (mcol, lambda x:  bootstrap_ci(x)[1])
                })

    smry =df.groupby(['model', 'variant']).agg(
        n_valid=('variant', 'size'),
        **metrics
    ).reset_index()
    return smry    

                     
class CombSaliencyCreator:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, me, inp, catidx):
        logging.info(f"CombSaliencyCreator: {len(self.inner)}")
        res = {}
        for inner in self.inner:
            ires = inner(me, inp, catidx)
            res.update(ires)
        return res
        

