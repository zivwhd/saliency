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





def get_dataset(me, dataset_name):
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    ds = Imagenet_Segmentation('/home/weziv5/work/data/gtsegs_ijcv.mat',
                            transform=me.get_transform(),
                            target_transform=test_lbl_trans)
    return ds



def get_creators_vit():
    baselines = [ZeroBaseline()]
    

    runs = [
        MultiCompExpCreator(desc="MWComp", segsize=[16], nmasks=[1000],  baselines = baselines,  groups=[
                            dict(c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150)
                                 ]),
        LTXSaliencyCreator(),
        IEMPertSaliencyCreator(),        
        RiseSaliencyCreator(),
        DimplVitSaliencyCreator()
    ]

    return CombSaliencyCreator(runs)


def get_creators_cnn():
    baselines = [ZeroBaseline()]
    

    runs = [
        MultiCompExpCreator(desc="MWComp", segsize=[40], nmasks=[500],  baselines = baselines,  groups=[
                            dict(c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150)
                                 ]),
        CamSaliencyCreator(),
        DixCnnSaliencyCreator(),
        IGSaliencyCreator(),                                 
        LTXSaliencyCreator(),
        IEMPertSaliencyCreator(),        
        RiseSaliencyCreator(),
    ]

    return CombSaliencyCreator(runs)

def get_creators(model_name):
    if 'resnet' in model_name or 'densenet' in model_name:
        return get_creators_cnn()
    if 'vit' in model_name:
        return get_creators_vit()
    assert False

def create_sals(model_name, dataset_name, marker="m"):
    me = ModelEnv(model_name)
    ds = get_dataset(me, dataset_name)
    progress_path = os.path.join("progress", model_name, f"create_{marker}")
    algo = get_creators(model_name)

    for idx, (img, tgt) in enumerate(ds):
        logging.info(f"[{idx}], {img.shape}, {tgt.shape}") ## torch.Size([3, 224, 224]), torch.Size([224, 224])
        coord = Coord([str(idx)], progress_path, getname=lambda x: str(x))
        for chk in coord:
            inp = img.to(me.device).unsqueeze(0)
            logits = me.model(inp).cpu()
            topidx = int(torch.argmax(logits))        
            logging.info(f"creating sal {idx} {topidx} ")

            sals = algo(me, inp, topidx)
            
            for variant, sal in sals.items():
                save_saliency(sal, model_name, variant, str(idx), run=0)

        if idx >= LIMIT_DS:
            logging.info("DONE")
            break

def eval_batch(Res, labels):
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1 - Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=0) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("outputs", outputs.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(outputs, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target

def create_scores(model_name, dataset_name):
    me = ModelEnv(model_name)
    ds = get_dataset(me, dataset_name)

    algo = get_creators()
    progress_path = os.path.join("progress", model_name, f"create_{marker}")

    for idx, (img, tgt) in enumerate(ds):

        res_path = "results/{model_name}/*/{idx}"
        sal_paths = glob.glob(res_path)

        result_prog = Coord(sal_paths, progress_path, getname=get_score_name)
        for path in result_prog:
            image_idx = os.path.basename(path)
            variant = os.path.basename(os.path.dirname(path))
            logging.debug(f"checking: {path} name={image_idx} variant={variant}")

            sal = torch.load(path)


            correct, labeled, inter, union, ap, f1, pred, target = eval_batch(
                torch.tensor(sal).unsqueeze(0), #.unsqueeze(0),
                torch.tensor(tgt).unsqueeze(0))

            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            total_ap += [ap]
            total_f1 += [f1]
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            mAp = np.mean(total_ap)
            mF1 = np.mean(total_f1)
            scores = {}
            scores[f'IoU'] = mIoU
            scores[f'mAP'] = mAp
            scores[f'pixAcc'] = pixAcc
            scores[f'mF1'] = mF1


        if idx >= LIMIT_DS:
            logging.info("DONE")
            break

def get_args(): 
    
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images", "create_sals", "scores", "summary", "all"], help="TBD")
    #parser.add_argument("--sal", choices=creators, default="cpe", help="TBD")
    parser.add_argument("--marker", default="m1", help="TBD")       
    parser.add_argument("--dataset", default="imagenet", help="TBD")       
    parser.add_argument("--selection", choices=["rsample3", "rsample10", "rsample100", "rsample1000", "rsample10K", "rsample5K", "show", "abl"], default="rsample3", help="TBD")       
    parser.add_argument("--model", choices=ALL_MODELS + ['all'], default="resnet50", help="TBD")    
    parser.add_argument('--ext', action='store_true', default=False, help="Enable extended mode")
    args = parser.parse_args()    
    return args


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
    logging.info("start")    
    args = get_args()

    args = get_args()
    model_name = args.model
    dataset_name = args.dataset
    if args.action == 'create_sals':
        create_sals(model_name, dataset_name, marker=args.marker)
    if args.action == 'scores':
        create_scores(model_name, dataset_name, marker=args.marker)



