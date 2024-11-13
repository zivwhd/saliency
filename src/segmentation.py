#!/bin/python

print("## segmentation.py ")

import argparse, logging, os, re
from dataset import ImagenetSource, Coord
from adaptors import CaptumCamSaliencyCreator, CamSaliencyCreator, METHOD_CONV
from adaptors_vit import AttrVitSaliencyCreator, DimplVitSaliencyCreator
from adaptors_gig import IGSaliencyCreator
from RISE import RiseSaliencyCreator
from cexcnn import CexCnnSaliencyCreator
from csixnn import IXNNSaliencyCreator
from acpe import TreSaliencyCreator
from benchmark import *
from cpe import *
from lcpe import CompExpCreator, MultiCompExpCreator, ZeroBaseline, RandBaseline, BlurBaseline
from mpert import IEMPertSaliencyCreator 
from extpert import ExtPertSaliencyCreator
from ltx import LTXSaliencyCreator
from dix_cnn import DixCnnSaliencyCreator
import torch
import socket
import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image, ImageFilter



LIMIT_DS = 1000

class Imagenet_Segmentation(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        # self.h5py = h5py.File(path, 'r+')
        self.h5py = None
        print(path)
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return min(self.data_length, LIMIT_DS)

VIT_MODELS = ["vit_small_patch16_224","vit_base_patch16_224","vit_base_patch16_224.mae"]
CNN_MODELS = ["resnet50","vgg16", "convnext_base", "densenet201"] ## "resnet18"
ALL_MODELS = CNN_MODELS + VIT_MODELS


def get_args(): 
    
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images", "create_sals", "scores", "summary", "all"], help="TBD")
    #parser.add_argument("--sal", choices=creators, default="cpe", help="TBD")
    parser.add_argument("--marker", default="m", help="TBD")       
    parser.add_argument("--dataset", default="imagenet", help="TBD")       
    parser.add_argument("--selection", choices=["rsample3", "rsample10", "rsample100", "rsample1000", "rsample10K", "rsample5K", "show", "abl"], default="rsample3", help="TBD")       
    parser.add_argument("--model", choices=ALL_MODELS + ['all'], default="resnet50", help="TBD")    
    parser.add_argument('--ext', action='store_true', default=False, help="Enable extended mode")
    args = parser.parse_args()    
    return args



def get_dataset(me, dataset_name):
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    ds = Imagenet_Segmentation('/home/weziv5/work/data/gtsegs_ijcv.mat',
                            transform=me.get_transform(),
                            target_transform=test_lbl_trans)
    return ds

def get_creators():
    baselines = [ZeroBaseline()]
    

    runs = [
        MultiCompExpCreator(desc="MWComp", segsize=[40], nmasks=[500],  baselines = baselines,  groups=[
                            dict(c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150)
                                 ]),
        CamSaliencyCreator(),
        DixCnnSaliencyCreator(),
        #IGSaliencyCreator(),                                 
        #LTXSaliencyCreator(),
        #IEMPertSaliencyCreator(),        
        #RiseSaliencyCreator(),
    ]

    return CombSaliencyCreator(runs)


def create_sals(model_name, dataset_name):
    me = ModelEnv(model_name)
    ds = get_dataset(me, dataset_name)

    algo = get_creators()

    for idx, (img, tgt) in enumerate(ds):
        logging.info(f"[{idx}], {img.shape}, {tgt.shape}")

        inp = img.to(me.device)
        logits = me.model(inp).cpu()
        topidx = int(torch.argmax(logits))        
        logging.info(f"creating sal {idx} {topidx} ")

        sals = algo(me, inp, topidx)
        
        for variant, sal in sals.items():
            save_saliency(sal, model_name, variant, str(idx), run=0)

        if idx >= LIMIT_DS:
            logging.info("DONE")
            break

def create_scores(model_name, dataset_name):
    me = ModelEnv(model_name)
    ds = get_dataset(me, dataset_name)

    algo = get_creators()

    for idx, (img, tgt) in enumerate(ds):

        res_path = "results/{model_name}/*/{idx}"
        sal_paths = glob.glob(res_path)

        if idx >= LIMIT_DS:
            logging.info("DONE")
            break


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
    logging.info("start")    
    args = get_args()

    args = get_args()
    model_name = args.model
    dataset_name = args.dataset
    create_sals(model_name, dataset_name)



