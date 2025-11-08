#!/bin/python

print("## segmentation.py ")
import logging
logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)

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
from lcpe import CompExpCreator, MultiCompExpCreator, AutoCompExpCreator, MulCompExpCreator, ZeroBaseline, RandBaseline, BlurBaseline, SegSlocExpCreator
from msm import MsmExpCreator
from mpert import IEMPertSaliencyCreator 
from extpert import ExtPertSaliencyCreator
from ltx import LTXSaliencyCreator
from dix_cnn import DixCnnSaliencyCreator
import torch
import socket
import os
from glob import glob
from collections import defaultdict
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image, ImageFilter
import sys

def setup_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    targets = [ os.path.join(os.path.dirname(current_dir),"LTX")  ]
    for path in targets:
        if path not in sys.path:
            logging.info(f"adding {path}")
            sys.path.append(path)

setup_path()
from utils.metrices import *

def dump_obj(obj, path):
    with open(path, "wb") as obf:
        pickle.dump(obj, obf)

LIMIT_DS = 1000
LIMIT_DS = 1697
#LIMIT_DS = 500

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
        #return self.data_length
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

    runs = [
        AutoCompExpCreator(
            desc="AutoComp", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
            epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        )
    ]

    return CombSaliencyCreator(runs)

    baselines = [ZeroBaseline()]
    #baselines = [RandBaseline()]

    basic = dict(#segsize=[16,48], nmasks=[500,500], 
                 c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=151, 
                 #select_from=10, select_freq=3, select_del=1.0,
                 select_from=None,
                 c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=1, c_model=0.0, c_norm=False, c_activation="")
    
    basic_mask_groups = {"Mix":{16:500,48:500}, "Sing":{48:1000}}

    def modify(**kwargs):
        args = basic.copy()
        args.update(**kwargs)
        return args

    lsc = MultiCompExpCreator(
            desc="LSC",
            mask_groups=basic_mask_groups,            
            baselines = baselines,
            pprob = [0.5],
            groups=[
                modify(),
            ])
    return lsc
    #return MultiCompExpCreator(desc="MYComp", segsize=[16], nmasks=[1000],  baselines = baselines,  groups=[
    #    dict(c_mask_completeness=1.0, c_magnitude=0.05, c_completeness=0, c_tv=0.7, c_model=0.0, c_norm=False, 
    #        c_activation="",  epochs=300, select_from=None)
    #        ]),

    return ExtPertSaliencyCreator()
    runs = [
        #MultiCompExpCreator(desc="MWComp", segsize=[16], nmasks=[1000],  baselines = baselines,  groups=[
        #                    dict(c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
        #                         c_activation="",  epochs=300, select_from=150)
        #                         ]),
        lsc,
        LTXSaliencyCreator(),
        IEMPertSaliencyCreator(),        
        RiseSaliencyCreator(),
        DimplVitSaliencyCreator()
    ]

    return CombSaliencyCreator(runs)



def get_creators_cnn():
    #return get_creators_vit()
    #return ExtPertSaliencyCreator()
    baselines = [ZeroBaseline()]    

    runs = [
        MultiCompExpCreator(desc="LSCx", 
                            mask_groups = {"Mix":{16:500,48:500}, "Sing":{48:1000}},  
                            baselines = baselines,  groups=[
                            dict(
                                c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=151, select_from=None,
                                c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, 
                                c_norm=False, c_activation="" )
                                 ]),
        #CamSaliencyCreator(),
        #DixCnnSaliencyCreator(),
        #IGSaliencyCreator(),                                 
        #LTXSaliencyCreator(),
        #IEMPertSaliencyCreator(),
        #ExtPertSaliencyCreator(),        
        #RiseSaliencyCreator(),        
    ]

    runs = [
        #AutoCompExpCreator(
        #    desc="AutoComp", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
        #    epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
        #    c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        #),
        #CamSaliencyCreator(),
        #DixCnnSaliencyCreator(),
        #IGSaliencyCreator(),                                 
        #LTXSaliencyCreator(),
        #IEMPertSaliencyCreator(),        
        #ExtPertSaliencyCreator(),        
        #RiseSaliencyCreator(),

    ]

    runs = [
        MultiCompExpCreator(desc="LSCx", 
                            mask_groups = {"S32":{32:1000}, "S48":{48:1000}},  pprob=[None, 0.5, 0.8],
                            baselines = baselines,  groups=[
                            dict(
                                c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=101, select_from=None,
                                c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=1, c_model=0.0, 
                                c_norm=False, c_activation="" )
                                 ]),
    ]   

    runs = [
        AutoCompExpCreator(
            desc="AutoCompSMix32x56", segsize=[32,56], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=15, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),

    ]

    baselines = [ZeroBaseline()]

    basic = dict(#segsize=[16,48], nmasks=[500,500], 
                 c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9, epochs=501, select_from=None,
                c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, c_activation="")

    runs = [
        DixCnnSaliencyCreator()
    ]
    
    runs_multi = [
        MultiCompExpCreator(
            desc="MULTSEG",
            pprob=[None],
            mask_groups={"m64":{64:1000}, "m56":{56:1000}, 
                         "m32":{32:1000}, "m32x64":{32:1000,64:1000},
                         "m32x64sm":{32:500,64:500}},
            baselines = baselines,
            groups=[
                basic
            ]),
            ExtPertSaliencyCreator(),
    ]

    return CombSaliencyCreator(runs)

def get_creators_abl():
    baselines = [ZeroBaseline()]

    basic = dict(#segsize=[16,48], nmasks=[500,500], 
                 c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=101, select_from=None,
                 #select_from=10, select_freq=3, select_del=1.0,                 
                c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, c_activation="")
    
    basic_mask_groups = {"":{32:500, 56:500}}
    def modify(**kwargs):
        args = basic.copy()
        args.update(**kwargs)
        return args
    runs = [
        MultiCompExpCreator(
            desc="LSCAbl",
            pprob=[None],
            mask_groups=basic_mask_groups,            
            baselines = baselines,
            groups=[

                modify(c_tv=0, desc="TVL"),
                modify(c_tv=0.01, desc="TVL"),
                modify(c_tv=0.1, desc="TVL"),
                modify(c_tv=0.2, desc="TVL"),
                modify(c_tv=0.5, desc="TVL"),
                modify(c_tv=1, desc="TVL"), ## 2
                
                modify(c_magnitude=0, desc="MAG"),
                modify(c_magnitude=0.01, desc="MAG"),
                modify(c_magnitude=0.05, desc="MAG"),
                modify(c_magnitude=0.1, desc="MAG"),
                modify(c_magnitude=0.25, desc="MAG"),                
                modify(c_magnitude=0.5, desc="MAG"),
                modify(c_magnitude=1, desc="MAG"), 
            ])
    ]
    return CombSaliencyCreator(runs)


def get_creators(model_name):
    
    #return MsmExpCreator(alphas=[0], blur_radius=[0])

    basic =  dict(
        desc="Auto", c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
        epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
        c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        force_desc=True
    )
    def modify(**kwargs):
        args = basic.copy()
        args.update(**kwargs)
        return args


    algo = MultiCompExpCreator(
        desc="Sloc",            
        mask_groups={
            f"Segs":{-20:200, -40:400, -60:400},
            f"MixLSeg":{-20:200, 56: 400,  32:400},
            f"MixMSeg":{-20:200, -40: 400,  56:400},
            f"MixSegSSq":{-20:200, -60: 300, 32:500}
        },
        pprob=[None],
        baselines = [ZeroBaseline()],
        groups=[modify(desc="GD"), modify(desc="Pos", c_magnitude=0, c_positive=1, c_tv=0.05)],
        acargs=dict(main_probs=[0.3,0.4,0.5,0.6,0.7,0.8], extra_probs=[], sampsize=40)
        )
    
    return algo

    return SegSlocExpCreator(
        desc = "SegMulti", seg_list=[(20,200,0.5),(40,400,0.5),(60,400,0.5)],
        c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
        epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
        c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",        
    )


def get_creators(model_name):
    basic =  dict(
        desc="Auto", c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
        epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 

        c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        force_desc=True
    )
    def modify(**kwargs):
        args = basic.copy()
        args.update(**kwargs)
        return args

    algo = MultiCompExpCreator(
        desc="Sloc",            
        mask_groups={
            f"RWLS":{ 32:500, 56:500},            
            #f"SegsHigh2":{-20:200, -40:400, -60:400},
            #f"SqSegsHigh":{-20:100, -40:200, -60:200, 32:250, 56:250},
        },
        pprob=[None],
        baselines = [ZeroBaseline()],
        groups=[modify(desc="LS", epochs=None, c_tv=100, c_magnitude=50 )],
        acargs=dict(main_probs=[0.3,0.4,0.5,0.6,0.7,0.8], extra_probs=[], sampsize=40)
        )
    return algo

def get_creators_(model_name):
    basic=dict(        
            desc="MSMLog", c_opt="Adam",
            select_from=None,  c_positive=0, 
            c_completeness=0, c_model=0.0, c_norm=False,  c_activation="",
            #force_desc=True,
            c_mask_completeness=1, c_magnitude=50, c_smoothness=0, c_tv=100, c_logistic=True,
            lr=0.001, lr_step=40, lr_step_decay=0.9, epochs=500
    )    

    def modify(**kwargs):
        args = basic.copy()
        args.update(**kwargs)
        return args

    return MultiCompExpCreator(
        desc="MsmLog",
        mask_groups={
            f"Ev3":{32:800, 56:200},
        },            
        baselines = [ZeroBaseline()],
        pprob=[None],
        acargs=dict(c_logistic=True),
        groups=[                    
            modify(),
                ])    
    
def get_creators_old(model_name):
    return get_creators_cnn()
    if 'resnet' in model_name or 'densenet' in model_name:
        return get_creators_cnn()
    if 'vit' in model_name:
        return get_creators_cnn()
        #return get_creators_vit()
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

    #ret = np.quantile(Res, 0.9)  ##Res.mean()
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

class Stats:
    def __init__(self):
        self.total_inter, self.total_union, self.total_correct, self.total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
        self.total_ap, self.total_f1 = [], []
        self.count = 0
    

def create_scores(model_name, dataset_name, marker="m"):
    me = ModelEnv(model_name)
    ds = get_dataset(me, dataset_name)

    #total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    #total_ap, total_f1 = [], []
    
    stats = defaultdict(Stats)
    #dump_obj(stats, f"results/{model_name}/bs.obj")
    progress_path = os.path.join("progress", model_name, f"scores_{marker}")

    variant_paths = glob(f"results/{model_name}/saliency/*")
    variant_list = [os.path.basename(x) for x in variant_paths]
    variant_list = [x for x in variant_list if not x.startswith('_')]
    for variant_name in Coord(variant_list, progress_path, getname=lambda x: str(x)):        
        for idx, (img, tgt) in enumerate(ds):
            logging.info(f"[{idx}], {img.shape}, {tgt.shape}") ## torch.Size([3, 224, 224]), torch.Size([224, 224])
            res_path = f"results/{model_name}/saliency/{variant_name}/{idx}"
            sal_paths = glob(res_path)
            assert len(sal_paths) <= 1
            if not sal_paths:
                continue
            path = sal_paths[0]
            logging.info(f"found: {res_path}: {sal_paths}")
        
        #result_prog = sal_paths #Coord(sal_paths, progress_path, getname=get_score_name)
            #for variant_name in Coord(list(variant_paths.keys()), progress_path, getname=lambda x: str(x)):
            #for path in variant_paths[variant_name]:
            if True:
                image_idx = os.path.basename(path)
                variant = os.path.basename(os.path.dirname(path))
                assert variant_name == variant_name
                if variant.startswith('_'):
                    continue
                logging.debug(f"checking: {path} name={image_idx} variant={variant}")
                vstat = stats[variant]
                
                sal = torch.load(path)

                correct, labeled, inter, union, ap, f1, pred, target = eval_batch(
                    torch.tensor(sal).unsqueeze(0), #.unsqueeze(0),
                    torch.tensor(tgt).unsqueeze(0))

                vstat.total_correct += correct.astype('int64')
                vstat.total_label += labeled.astype('int64')
                vstat.total_inter += inter.astype('int64')
                vstat.total_union += union.astype('int64')
                vstat.total_ap += [ap]
                vstat.total_f1 += [f1]
                vstat.count += 1

                logging.info(f"::STATS,{image_idx},{variant},{int(correct.astype('int64'))},{int(labeled.astype('int64'))},{int(inter.astype('int64')[0])},{int(union.astype('int64')[0])},{int(inter.astype('int64')[1])},{int(union.astype('int64')[1])}")
                ###
                #pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
                #IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
                #mIoU = IoU.mean()
                #mAp = np.mean(total_ap)
                #mF1 = np.mean(total_f1)
                #scores = {}
                #scores[f'IoU'] = mIoU
                #scores[f'mAP'] = mAp
                #scores[f'pixAcc'] = pixAcc
                #scores[f'mF1'] = mF1


            if idx >= LIMIT_DS:
                logging.info("DONE")
                break
            #dump_obj(stats, f"results/{model_name}/stats.obj")
        stats_path = f"results/{model_name}/stats/{variant_name}"
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "wt") as sf:
            write_stats({variant_name : stats[variant_name]}, sf)

def write_stats(stats, out):
    print(f'variant,nsamples,mIoU,mAp,pixAcc,mF1', file=out)
    for variant, vstat in stats.items():
        pixAcc = np.float64(1.0) * vstat.total_correct / (np.spacing(1, dtype=np.float64) + vstat.total_label)
        IoU = np.float64(1.0) * vstat.total_inter / (np.spacing(1, dtype=np.float64) + vstat.total_union)
        mIoU = IoU.mean()
        mAp = np.mean(vstat.total_ap)
        mF1 = np.mean(vstat.total_f1)

        print(f'{variant},{vstat.count},{mIoU},{mAp},{pixAcc},{mF1}', file=out)

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



