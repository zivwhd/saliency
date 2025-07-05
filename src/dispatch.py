#!/bin/python

print("## dispatch.py ")

import argparse, logging, os, re
from dataset import ImagenetSource, VOCSource, Coord
from adaptors import CaptumCamSaliencyCreator, CamSaliencyCreator, METHOD_CONV
from adaptors_vit import AttrVitSaliencyCreator, DimplVitSaliencyCreator
from adaptors_gig import IGSaliencyCreator
from RISE import RiseSaliencyCreator
from cexcnn import CexCnnSaliencyCreator
from csixnn import IXNNSaliencyCreator
from acpe import TreSaliencyCreator
from benchmark import *
from cpe import *
from lcpe import CompExpCreator, MultiCompExpCreator, AutoCompExpCreator, MProbCompExpCreator, ZeroBaseline, RandBaseline, BlurBaseline, MulCompExpCreator
from mpert import IEMPertSaliencyCreator 
from extpert import ExtPertSaliencyCreator
from ltx import LTXSaliencyCreator
from dix_cnn import DixCnnSaliencyCreator
from sanity import SanityCreator
import torch
import socket



def get_dixcnn_sal_creator():
    return DixCnnSaliencyCreator()

def get_mpert_sal_creator():
    return IEMPertSaliencyCreator()

def get_extpert_sal_creator():
    return ExtPertSaliencyCreator()

def get_ltx_sal_creator():
    return LTXSaliencyCreator()

def get_cpltx_sal_creator():
    cp_gen = CompExpCreator(nmasks=500, segsize=48)
    return LTXSaliencyCreator(cp_gen=cp_gen)

def get_sanity_sal_creator():
    return SanityCreator()

def get_sanity_ext_sal_creator():
    return SanityCreator(nmasks=2000, c_magnitude=0.1)

def get_mrcomp_cnn_sal_creator():
    ## desc="MrCompA", segsize=[32,40,48], nmasks=[300,400,300],
    return CompExpCreator(
        desc="MrCompA", segsize=[32,40,48], nmasks=[300,400,300],
        c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
        c_activation="",  epochs=300, select_from=150
    )

def get_mrcomp_chk_sal_creator():
    ## desc="MrCompA", segsize=[32,40,48], nmasks=[300,400,300],
    return CompExpCreator(
        desc="MrCompKl", segsize=[48,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, 
        epochs=100, select_from=None, select_freq=3, select_del=1.0,
        c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
        c_activation="",
    )

def get_mrcomp_vit_sal_creator():
    return CompExpCreator(
        desc="MrCompLv", segsize=[16,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, 
        epochs=301, select_from=200, select_freq=3, select_del=1.0,
        c_mask_completeness=1.0, c_magnitude=1, c_positive=True, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
        c_activation="",
    )

def get_mrcomp_sal_creator():
    ## desc="MrCompA", segsize=[500,250,250], nmasks=[16,24,32],
    ## desc="MrCompB", segsize=[16,48], nmasks=[500,500],
    ## desc="MrCompC", segsize=[16,48], nmasks=[500,500], lr=0.01, epochs=500, ## this one is good
    ## desc="MrCompD", segsize=[16,48], nmasks=[500,500], lr=0.01, epochs=500, c_opt="AdamW",
    ## desc="MrCompE", segsize=[16,48], nmasks=[500,500],  c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=101, select_from=10, select_freq=3,
    ## desc="MrCompF", segsize=[32,40,48], nmasks=[300,400,300], c_opt="Adam", lr=4, lr_step=9, lr_step_decay=0.9, epochs=101, select_from=10, select_freq=3, c_logit=True,
    ## desc="MrCompG", segsize=[16,48], nmasks=[500,500], c_opt="Adam", lr=4, lr_step=9, lr_step_decay=0.9, epochs=101, select_from=10, select_freq=3, c_logit=True,
    ## desc="MrCompH", segsize=[16,48], nmasks=[300,700],  c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=101, select_from=10, select_freq=3,
    ## desc="MrCompI", segsize=[16,48], nmasks=[700,300], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=101, select_from=10, select_freq=3, ## select-freq didn't pass
    ## desc="MrCompJ", segsize=[16,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=101, select_from=10, select_freq=3, select_del=1.0,
    ## desc="MrCompK", segsize=[16,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=151, select_from=10, select_freq=3, select_del=1.0, pprob=0.1,c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
    ## desc="MrCompL", segsize=[48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=301, select_from=200, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=1, c_positive=True, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
    return CompExpCreator(
        desc="MrCompM", segsize=[16,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, 
        epochs=101, select_from=10, select_freq=3, select_del=1.0,
        c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
        c_activation="",
    )

def get_altcomp_sal_creator():
    return AutoCompExpCreator(
        desc="AutoComp", segsize=[32], nmasks=[4000], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
        epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.1, c_positive=0, 
        c_completeness=0, c_tv=1.0, c_model=0.0, c_norm=False,  c_activation="", cap_response=False
    )

def get_autocomp_sal_creator():
    # desc="AutoCompA",  NOT GOOD segsize=32, nmasks=1000, c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  
    # desc="AutoCompB", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  epochs=301, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
    # desc="AutoCompC", segsize=[16,32,48], nmasks=[300,400,300], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=301, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
    #desc="AutoCompD", segsize=[16,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=201, select_from=10, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
    #desc="AutoCompE", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  epochs=301, select_from=10, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
    
    return CombSaliencyCreator([
        AutoCompExpCreator(
            desc="AutoComp", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
            epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        AutoCompExpCreator(
            desc="AutoCompMon", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
            epochs=101, select_from=10, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        MProbCompExpCreator(
            desc="MProbComp", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
            epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),        
        ])

def get_autoncomp_sal_creator():

    return CombSaliencyCreator([
        MultiCompExpCreator(
            desc="SLOC",            
            mask_groups={f"Mix":{32:500, 56:500}},
            pprob=[None],
            baselines = [ZeroBaseline()],
            groups=[
                dict(
                    desc="Auto", c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
                    epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
                    c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
                ),
                dict(
                    desc="AutoMon", c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
                    epochs=501, select_from=50, select_freq=15, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
                    c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",                    
                )
            ]
        ),
        MProbCompExpCreator(
            desc="SLOCProb", segsize=[32,56], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
            epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),

    ])

                


def get_ocomp_sal_creator():    
    return CombSaliencyCreator([
        #CompExpCreator(
        #    desc="CompBs5", segsize=[32], nmasks=[1000], pprob=[0.5], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
        #    epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
        #    c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        #),
        #CompExpCreator(
        #    desc="CompEs5", segsize=[32], nmasks=[1000], pprob=[0.5], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
        #    epochs=101, select_from=10, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
        #    c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        #),
        #CompExpCreator(
        #    desc="CompBs2", segsize=[32], nmasks=[1000], pprob=[0.2], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
        #    epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
        #    c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        #),
        #CompExpCreator(
        #    desc="CompEs2", segsize=[32], nmasks=[1000], pprob=[0.2], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
        #    epochs=101, select_from=10, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
        #    c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        #),
        CompExpCreator(
            desc="CompBs8", segsize=[32], nmasks=[1000], pprob=[0.8], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
            epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        CompExpCreator(
            desc="CompEs8", segsize=[32], nmasks=[1000], pprob=[0.8], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
            epochs=101, select_from=10, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        #CompExpCreator(
        #    desc="CompJb", segsize=[16,48], nmasks=[500,500], pprob=[0.5,0.5], c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  
        #    epochs=101, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
        #    c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        #)
        ])


def get_lscnm_sal_creator():

    return MultiCompExpCreator(
        desc="LSC",

        mask_groups={f"xL":{48:500}, f"xS":{48:1000},},            
            baselines = [ZeroBaseline()],
            groups=[
                dict(c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  epochs=101, select_from=None, 
                     c_mask_completeness=1.0, c_magnitude=0.01, c_positive=True, c_completeness=0, c_tv=0.1, 
                     c_model=0.0, c_norm=False,  c_activation=""),
                dict(c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9,  epochs=101, select_from=None, 
                     c_mask_completeness=1.0, c_magnitude=0.01, c_positive=False, c_completeness=0, c_tv=0.1, 
                     c_model=0.0, c_norm=False,  c_activation=""),

            ])


def get_epochcheck_sal_creator():
    baselines = [ZeroBaseline()]
    basic_mask_groups = {"":{16:500, 48:500}}

    basic = dict(
                 c_opt="Adam", lr=0.1, lr_step=9, lr_step_decay=0.9, epochs=101, 
                 select_from=None,#select_from=10, select_freq=3, select_del=1.0,
                 c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, c_activation="")
    def modify(**kwargs):
        args = basic.copy()
        args.update(**kwargs)
        return args
    
    return MultiCompExpCreator(
            desc="SEG",
            mask_groups=basic_mask_groups,
            baselines = baselines,
            groups=[
                modify(epochs=x) for x in [fac*10 for fac in range(1,30)]
            ])


def get_prob_sal_creator(nmasks=1000):
    basic = dict(
        c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9, epochs=501, select_from=None,
        c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, c_activation="")
        
    return MultiCompExpCreator(
            desc="PROB",
            pprob=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            mask_groups={"m32":{32:750}, "m56":{56:750} },
            baselines = [ZeroBaseline()],
            groups=[
                dict(
                    c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9, epochs=501, select_from=None,
                    c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, c_activation="")
            ])

def get_abl2_sal_creator(nmasks=1000):

    logging.info("ablation sals")
    baselines = [ZeroBaseline()]

    basic = dict(#segsize=[16,48], nmasks=[500,500], 
                 c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9, epochs=501, select_from=None,
                c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, c_activation="")
    

    runs = [
        MulCompExpCreator(
            desc="MinRomp32", segsize=[[32]]*10, mode="min", nmasks=1000, c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01,
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        )
    ]

    runs_ext = [
        AutoCompExpCreator(
            desc="AutoCompSMix32x56", segsize=[32,56], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        AutoCompExpCreator(
            desc="AutoCompSMixMon32x56", segsize=[32,56], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=50, select_freq=15, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        AutoCompExpCreator(
            desc="AutoCompSMix32x48", segsize=[32,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        AutoCompExpCreator(
            desc="AutoCompSMixMon32x48", segsize=[32,48], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=50, select_freq=15, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        AutoCompExpCreator(
            desc="AutoCompSMixPos", segsize=[32,56], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0, c_positive=1, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        MulCompExpCreator(
            desc="MulComp32x56", segsize=[32,56], nmasks=1000, c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01,
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),


    ]
    runs_ext = [
        AutoCompExpCreator(
            desc="AutoCompSlow", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        AutoCompExpCreator(
            desc="AutoCompSlowMon", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=100, select_freq=15, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),

        MulCompExpCreator(
            desc="MulComp32x32", segsize=[32,32], nmasks=1000, c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01,
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),        
        MulCompExpCreator(
            desc="MulComp32x56", segsize=[32,56], nmasks=1000, c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01,
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),

    AutoCompExpCreator(
            desc="AutoCompNoL1", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),        
    AutoCompExpCreator(
            desc="AutoCompNoReg", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0, c_positive=0, 
            c_completeness=0, c_tv=0, c_model=0.0, c_norm=False,  c_activation="",
        ),        
        AutoCompExpCreator(
            desc="AutoCompVSlow", segsize=[32], nmasks=[1000], c_opt="Adam", lr=0.1, lr_step=90, lr_step_decay=0.9,  
            epochs=1001, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        MultiCompExpCreator(
            desc="MULTSEG",
            pprob=[None],
            mask_groups={"m16x48":{16:500, 48:500}, "m32x64":{32:500, 56:500}, 
                         "m32xx56":{32:1000, 56:500}, "m16to56":{16:500, 32:500, 48:500, 56:500}},
            baselines = baselines,
            groups=[
                basic
            ])
    ]
    return CombSaliencyCreator(runs)

def get_abl_sal_creator(nmasks=1000):

    logging.info("ablation sals")
    baselines = [ZeroBaseline()]

    basic = dict(#segsize=[16,48], nmasks=[500,500], 
                 c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9, epochs=501, select_from=None,
                 #select_from=10, select_freq=3, select_del=1.0,                 
                c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, c_activation="")
    
    basic_mask_groups = {"":{32:1000, 56:1000}}

    def modify(**kwargs):
        args = basic.copy()
        args.update(**kwargs)
        return args

    
    runs = [
        MultiCompExpCreator(
            desc="SEG",
            pprob=[None],
            mask_groups={f"_{seg}_":{seg:1000} for seg in [8,16,32,40,48,56,64]},            
            baselines = baselines,
            groups=[
                basic
            ]),

        MultiCompExpCreator(
            desc="MSK",
            pprob=[None],
            mask_groups={f"_{nmasks}_": { 32 : int(nmasks/2), 56 : int(nmasks/2) } for nmasks in [10,100,250,500,750,1000, 1250, 1500, 2000]},
            baselines = baselines,
            groups=[
                basic,
            ]),

        MultiCompExpCreator(
            desc="BSLN",
            pprob=[None],
            mask_groups=basic_mask_groups,
            baselines = [ZeroBaseline(),BlurBaseline(),RandBaseline()],
            groups=[basic]),

        MultiCompExpCreator(
            desc="PROB",
            pprob=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            mask_groups=basic_mask_groups,
            baselines = baselines,
            groups=[basic]),

        MultiCompExpCreator(
            desc="MComp",
            pprob=[None],
            mask_groups=basic_mask_groups,            
            baselines = baselines,
            groups=[
                #modify(desc="CPONLY", c_tv=0, c_magnitude=0),        
                #modify(c_model=0.01, desc="MDLA", c_mask_completeness=0),
                #modify(c_model=0.05, desc="MDLA", c_mask_completeness=0),

                modify(c_mask_completeness=0, desc="NCP"),
                modify(c_mask_completeness=0.01, desc="NCP"),
                modify(c_mask_completeness=0.1, desc="NCP"),
                modify(c_mask_completeness=1.0, desc="NCP"),

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

                modify(epochs=25, lr_step=2, desc="EPC"),
                modify(epochs=50, lr_step=4, desc="EPC"),
                modify(epochs=75, lr_step=6, desc="EPC"),
                modify(epochs=100, lr_step=9, desc="EPC"),                
                modify(epochs=200, lr_step=18, desc="EPC"),
                modify(epochs=250, lr_step=22, desc="EPC"),
                modify(epochs=500, lr_step=45, desc="EPC"),
                modify(epochs=750, lr_step=67, desc="EPC"),
                modify(epochs=1000, lr_step=90, desc="EPC"),
                
                #modify(epochs=100),
                #modify(epochs=100, select_from=None, desc="MNT"),
                #modify(epochs=200, select_from=None, desc="MNT"),
                #modify(epochs=300, select_from=None, desc="MNT"),
                #modify(epochs=400, select_from=None, desc="MNT"),
                ##modify(epochs=500, select_from=None, desc="MNT"),
                
                #modify(c_model=0, desc="MDL"),
                #modify(c_model=0.01, desc="MDL"),
                #modify(c_model=0.05, desc="MDL"),
                #modify(c_model=0.1, desc="MDL"),
                #modify(c_model=0.2, desc="MDL"), 
            ])
    ]
    return CombSaliencyCreator(runs)


def get_mwcomp_cnn_sal_creator():

    baselines = [ZeroBaseline()]
    return CompExpCreator(
        desc="MWComp", segsize=[40], nmasks=[500], c_opt="Adam", epochs=300, select_from=150,
        c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
        c_activation="")

    return MultiCompExpCreator(desc="MWComp", segsize=[40], nmasks=[500],  baselines = baselines, 
                        groups=[
                            dict(c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150),
                        ])


def get_mwcomp_vit_sal_creator():
    baselines = [ZeroBaseline()]
    return CompExpCreator(
        desc="MWComp", segsize=[16], nmasks=[1000], c_opt="Adam", epochs=300, select_from=150,
        c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
        c_activation="")

    return MultiCompExpCreator(desc="MWComp", segsize=[24, 32, 40], nmasks=[1000],  baselines = baselines, 
                        groups=[
                            dict(c_mask_completeness=1.0, c_magnitude=0.01, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150),
                        ])


def get_mwcomp_ext_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(desc="MWComp", segsize=[40], nmasks=[2000],  baselines = baselines, 
                        groups=[
                            dict(c_mask_completeness=1.0, c_magnitude=0.1, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150),
                        ])

def get_mwcomp_vis_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(desc="MWComp", segsize=[40], nmasks=[500,2000],  baselines = baselines, 
                        groups=[
                            dict(c_mask_completeness=1.0, c_magnitude=0.1, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150),
                            dict(c_mask_completeness=1.0, c_magnitude=0.5, c_completeness=0, c_tv=0.5, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150)
                        ])


    
def get_mwcomp_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(desc="MWComp", segsize=[16,32,40], nmasks=[500,1000],  baselines = baselines, 
                        groups=[
                            dict(c_mask_completeness=1.0, c_magnitude=0.05, c_completeness=0, c_tv=0, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=100),
                            dict(c_mask_completeness=1.0, c_magnitude=0.1, c_completeness=0, c_tv=0, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=100),
                            dict(c_mask_completeness=1.0, c_magnitude=0.5, c_completeness=0, c_tv=0, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=100),
                                 
                            dict(c_mask_completeness=1.0, c_magnitude=0.05, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="", epochs=300, select_from=150),
                            dict(c_mask_completeness=1.0, c_magnitude=0.1, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150),
                            dict(c_mask_completeness=1.0, c_magnitude=0.5, c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False, 
                                 c_activation="", epochs=300, select_from=150),
                                 
                            dict(c_mask_completeness=1.0, c_magnitude=0.05, c_completeness=0, c_tv=0.3, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150),
                            dict(c_mask_completeness=1.0, c_magnitude=0.1, c_completeness=0, c_tv=0.3, c_model=0.0, c_norm=False, 
                                 c_activation="", epochs=300, select_from=150),
                            dict(c_mask_completeness=1.0, c_magnitude=0.5, c_completeness=0, c_tv=0.3, c_model=0.0, c_norm=False, 
                                 c_activation="",  epochs=300, select_from=150)
                                 ]
                            )

    
def get_obench_sal_creator():
    return CombSaliencyCreator([
        AutoCompExpCreator(
            desc="AutoCompSMix32x56", segsize=[32,56], nmasks=[500,500], c_opt="Adam", lr=0.1, lr_step=45, lr_step_decay=0.9,  
            epochs=501, select_from=None, select_freq=3, select_del=1.0, c_mask_completeness=1.0, c_magnitude=0.01, c_positive=0, 
            c_completeness=0, c_tv=0.1, c_model=0.0, c_norm=False,  c_activation="",
        ),
        DimplVitSaliencyCreator(),
        LTXSaliencyCreator(),        
        ExtPertSaliencyCreator(),
        RiseSaliencyCreator(),
        IEMPertSaliencyCreator(),

    ])

def get_mbench_sal_creator():
    baselines = [ZeroBaseline()]
    
    runs  = [
        MultiCompExpCreator(desc="MComp", segsize=[40], nmasks=msk,  baselines = baselines, 
                            groups=[dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.3, c_model=0.0, c_norm=True, c_activation="", epochs=10)]
                            )
        for msk in [50, 100, 500, 1000, 1500]
        #MultiCompExpCreator(desc="MComp", segsize=[40], nmasks=[50, 100, 500, 1000, 1500, 2000],  baselines = baselines, 
        #                    groups=[
        #                        dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.3, c_model=0.0, c_norm=True, c_activation="", epochs=epochs)
        #                        for epochs in [50, 100, 200, 300, 400]
        #                        ] + [
        #                        dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.3, c_model=0.1, c_norm=True, c_activation="", epochs=epochs)
        #                        for epochs in [50, 100, 200, 300, 400]
        #                        ] 
        #                    )
    ]
    return CombSaliencyCreator(runs)
    

def get_ablvit_sal_creator():
    return get_abl_sal_creator(segsize=16, nmasks=1000)


def get_mcompvitD_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(
        desc="MComp",
        segsize=[16], nmasks=1000,
        baselines = baselines,
        groups=[
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.05, c_norm=True, c_activation=None),             
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.1, c_norm=True, c_activation=None)
            ] )

def get_mcompD_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(
        desc="MComp",
        segsize=[40], nmasks=500, 
        baselines = baselines,
        groups=[
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation=""), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.2, c_model=0, c_norm=True, c_activation=""), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.3, c_model=0.05, c_norm=True, c_activation=""), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.4, c_model=0, c_norm=True, c_activation=""),             
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.5, c_model=0, c_norm=True, c_activation=""), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=1, c_model=0, c_norm=True, c_activation=""), 

            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.2, c_model=0.05, c_norm=True, c_activation=""), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation=""), 

            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.3, c_model=0, c_norm=False, c_activation=None, c_magnitude=0), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.3, c_model=0, c_norm=False, c_activation=None, c_magnitude=0), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.05, c_model=0.1, c_norm=True, c_activation="sigmoid", c_magnitude=0), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.5, c_model=0, c_norm=True, c_activation="sigmoid", c_magnitude=0), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.15, c_model=0.1, c_norm=True, c_activation="sigmoid", c_magnitude=0), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation="sigmoid", c_magnitude=0.05),
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation="sigmoid", c_magnitude=0.02),
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation="sigmoid", c_magnitude=-0.02),
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation="sigmoid", c_magnitude=-0.05),
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation="sigmoid", c_magnitude=-0.1),
    ])


def get_mcomp2_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(
        desc="MComp",
        segsize=[32, 48], nmasks=500, 
        baselines = baselines,
        groups=[
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.05, c_norm=True, c_activation="sigmoid"), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.15, c_model=0.05, c_norm=True, c_activation=""),
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.05, c_norm=True, c_activation=""), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.05, c_model=0.05, c_norm=True, c_activation=""), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.02, c_model=0.05, c_norm=True, c_activation=""), 

            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.05, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.1), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.05, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.2), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.05, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.5), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.1), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.2), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.5), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.1), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.2), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0, c_model=0.05, c_norm=True, c_activation="", c_magnitude=0.5), 
                        
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.1, c_norm=True, c_activation="sigmoid"), 
        ])

def get_mcomp5_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(
        desc="MComp",
        segsize=[16, 24], nmasks=1000, 
        baselines = baselines,
        groups=[
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.1, c_model=0, c_norm=False, c_activation="sigmoid"), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation=""),
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.2, c_model=0, c_norm=True, c_activation=""),
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.5, c_model=0, c_norm=True, c_activation=""), 
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=1, c_model=0, c_norm=True, c_activation=""),             
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=1.5, c_model=0, c_norm=True, c_activation=""),             
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.05, c_model=0, c_norm=False, c_activation="", c_magnitude=0.1), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.05, c_model=0, c_norm=False, c_activation="", c_magnitude=0.2), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.05, c_model=0, c_norm=False, c_activation="", c_magnitude=0.5), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.1, c_model=0, c_norm=False, c_activation="", c_magnitude=0.1), 
            ##dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.1, c_model=0, c_norm=False, c_activation="", c_magnitude=0.2), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0.1, c_model=0, c_norm=False, c_activation="", c_magnitude=0.5), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0, c_model=0, c_norm=False, c_activation="", c_magnitude=0.1), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0, c_model=0, c_norm=False, c_activation="", c_magnitude=0.2), 
            #dict(c_mask_completeness=1.0, c_completeness=0.1, c_tv=0, c_model=0, c_norm=False, c_activation="", c_magnitude=0.5), 
                        
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.1, c_norm=True, c_activation="sigmoid"), 
        ])

def get_mcompvit_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(
        desc="MComp",
        segsize=[32], nmasks=2000, 
        baselines = baselines,
        groups=[
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation="sigmoid"),             
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.1, c_norm=True, c_activation="sigmoid"), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation="tanh"), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.15, c_model=0, c_norm=True, c_activation="sigmoid"),             
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.05, c_model=0, c_norm=True, c_activation="sigmoid"), 
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.02, c_model=0, c_norm=True, c_activation="sigmoid"), 

    ])

def get_mcompvitA_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(
        desc="MComp",
        segsize=[16], nmasks=1000,
        baselines = baselines,
        groups=[
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0, c_norm=True, c_activation=None),             
            #dict(c_mask_completeness=1.0, c_completeness=0, c_tv=0.1, c_model=0.1, c_norm=True, c_activation=None)
            ] )
            

def get_mcompvitB_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(
        desc="MComp",
        segsize=[16], nmasks=1000,
        baselines = baselines,
        groups=[
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=1, c_model=0, c_norm=True, c_activation=None),             
            dict(c_mask_completeness=1.0, c_completeness=0, c_tv=1, c_model=0.05, c_norm=True, c_activation=None)
            ] )


def get_emask_sal_creator():
    baselines = [ZeroBaseline()]
    return MultiCompExpCreator(
        desc="EMask",
        segsize=[48], nmasks=500, 
        baselines = baselines,
        groups=[        
            dict(c_mask_completeness=0, c_completeness=0.1, c_model=0.5, c_tv=2, c_magnitude=2),
            dict(c_mask_completeness=0.1, c_completeness=0.1, c_model=0.5, c_tv=2, c_magnitude=2),
    ])

def get_comp_sal_creator():

    return CompExpCreator(
        segsize=64, avg_kernel_size=(17,17), 
        nmasks=1000, epochs=150, 
        c_mask_completeness=1, c_completeness=0.1, c_smoothness=0.5,        
        c_selfness=0,
                      )    

    return CompExpCreator(segsize=48, avg_kernel_size=(17,17), epochs=700, 
                      c_mask_completeness=1, c_completeness=0.1, c_smoothness=0.5,
                      c_selfness=0,
                      desc="CompRd"
                      )
    
    ## 9.228502593248116,54.224837325563946
    return CompExpCreator(
        segsize=48, avg_kernel_size=(21,21), epochs=500, nmasks=500,
        c_mask_completeness=15, c_completeness=1, c_smoothness=2,
        c_selfness=0, desc="CompR")
    return CompExpCreator(avg_kernel_size=(2,2), epochs=500, beta=1, alpha=1)
    #return CompExpCreator()

def get_tre_sal_creator(limit=100):
    return TreSaliencyCreator(limit)

def get_cpe_sal_creator(segsize=64):
    return IpwSalCreator(f"CPE_{segsize}", [10, 100, 250, 500,1000,2000,4000], segsize=segsize, batch_size=32)

def get_pcpe_sal_creator(segsize=64):
    return IpwSalCreator(f"PCPE_{segsize}", [10, 100, 250, 500, 1000, 2000,4000], segsize=segsize, with_softmax=True, batch_size=32)

def get_pcpe_abl_clip_sal_creator(segsize=64):
    return IpwSalCreator(f"ABLC_{segsize}", [2000, 4000], clip=[0, 0.01, 0.1, 0.2, 0.25, 0.5, 0.75, 1], segsize=segsize, with_softmax=True, batch_size=32)

def get_pcpe_abl_out_sal_creator(segsize=64):
    return AblIpwSalCreator(f"ABLO_{segsize}", [2000, 4000], clip=[0.1], segsize=segsize, batch_size=32)

def get_pcpe_abl_samp_sal_creator(segsize=64):
    return IpwSalCreator(f"ABLS_{segsize}", [10, 100, 250, 500, 1000, 2000, 3000, 4000, 5000], segsize=segsize, with_softmax=True, batch_size=32)

def get_pcpe_abl_seg_sal_creator():
    inner = [
        IpwSalCreator(f"ABL_{segsize}", [2000], segsize=segsize, with_softmax=True, batch_size=32)
        for segsize in [20, 32, 48,64,80]
    ]
    return CombSaliencyCreator(inner)


def get_rcpe_sal_creator(segsize=64):
    return IpwSalCreator(f"RCPE_{segsize}", [500,1000,2000,4000], segsize=segsize, batch_size=32, ipwg=RelIpwGen)

def get_cam_sal_creator():
    return CamSaliencyCreator()

def get_tattr_sal_creator():
    return AttrVitSaliencyCreator()

def get_dimpl_sal_creator():
    return DimplVitSaliencyCreator()

def get_dix_sal_creator():
    return DimplVitSaliencyCreator(['dix'])

def get_rise_sal_creator():
    return RiseSaliencyCreator()

def get_captum_sal_creator():
    return CaptumCamSaliencyCreator()

def get_gig_sal_creator():
    return IGSaliencyCreator()

def get_cex_sal_creator():
    return CexCnnSaliencyCreator()

def get_ixnn_sal_creator():
    return IXNNSaliencyCreator()

ALL_CNN_CREATORS = ["pcpe", "rise", "cam", "cex", "gig" ]
ALL_VIT_CREATORS = ["pcpe", "rise", "dimpl"]

def create_sals_by_name(names, me, images, marker="c1"):
    if type(names) == str:
        names = [names]

    for name in names:
        logging.info(f"create sals: {name}")
        progress_path = os.path.join("progress", me.arch, f"create_sals_{name}_{marker}")
        coord_images = Coord(images, progress_path)

        cname = f"get_{name}_sal_creator"
        func = globals()[cname]
        algo = func()
        create_saliency_data(me, algo, coord_images, run_idx=0)

def create_model_sals(model_name, sal_names, marker="c1"):
    me = ModelEnv(model_name)    
    if sal_names == 'all':
        if model_name in CNN_MODELS:
            sal_names = ALL_CNN_CREATORS
        elif model_name in VIT_MODELS:
            sal_names = ALL_VIT_CREATORS
        else:
            assert False
    create_sals_by_name(sal_names, me, all_images, marker=marker)

def include_result(x):
    if x.startswith('_'):
        return False
    if (("CPE_" in x) and ("_4000_" not in x) and ("_2000_" not in x) and ("_500_" not in x)):
        return False
    if ('CexCnn_' in x) and ('_0.75_' not in x) and ('_0.95_' not in x) and ('_0.995_' not in x):
        return False
    if ('CexCnnA_' in x) and  ('_0.5_' not in x) and ('_0.75_' not in x) and ('_0.95_' not in x) and ('_0.995_' not in x):
        return False
    if ('ExtPert' in x) and ('ExtPertM' not in x):
        return False
    
    return True

def create_model_scores(model_name, marker="c1", extended=False, equant=False, subset=None):
    me = ModelEnv(model_name)            
    result_paths = get_all_results(model_name, subset=subset)
    result_paths = [x for x in result_paths if include_result(x)]
    logging.info(f"found {len(result_paths)} saliency maps")
    ext_mark = get_ext_mark(extended=extended, equant=equant)
    
    progress_path = os.path.join("progress", model_name, f"{ext_mark}scores_any_{marker}")
    result_prog = Coord(result_paths, progress_path, getname=get_score_name)            
    create_scores(me, result_prog, all_images_dict, update=True, extended=extended, equant=equant)

def create_model_summary(model_name, extended=False, equant=False):
    logging.info(f"summary for {model_name} {extended} {equant}")
    emark = get_ext_mark(extended=extended, equant=equant)
    base_csv_path = os.path.join("results", model_name)
    df = load_scores_df(model_name, filter_func=include_result, extended=extended, equant=equant)
    df.to_csv(f'{base_csv_path}/{emark}results.csv', index=False)
    smry = summarize_scores_df(df, extended=extended, equant=equant)
    smry.to_csv(f'{base_csv_path}/{emark}summary.csv', index=False)
    return smry

def create_model_summary_all(model_name):
    main = create_model_summary(model_name)
    ext = create_model_summary(model_name, extended=True)
    merged = pd.merge(main, ext, on=['model', 'variant'], how='left')
    base_csv_path = os.path.join("results", model_name)
    merged.to_csv(f'{base_csv_path}/all_summary.csv', index=False)



def get_creators():
    ptrn = re.compile("get_(.*)_sal_creator")
    return [match.group(1) for match in [ptrn.match(vr) for vr in globals()] if match is not None]

VIT_MODELS = ["vit_small_patch16_224","vit_base_patch16_224","vit_base_patch16_224.mae"]
CNN_MODELS = ["resnet50","vgg16", "convnext_base", "densenet201","densenet201NT","densenet201RT","resnet50NT", "resnet50RT", "vgg16NT", "vgg16RT"] ## "resnet18"
ALL_MODELS = CNN_MODELS + VIT_MODELS

def get_args(): 
    creators = get_creators() + ['any','all']
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images", "create_sals", "scores", "summary", "asummary", "all"], help="TBD")
    parser.add_argument("--sal", choices=creators, default="cpe", help="TBD")
    parser.add_argument("--dataset", choices=["imagenet","voc"], default="imagenet", help="TBD")
    parser.add_argument("--marker", default="m", help="TBD")       
    parser.add_argument("--selection", choices=["snty","vis", "rsample3", "rsample10", "rsample100", "rsample500", "rsample1000", "rsample2500", "rsample10K", "rsample5K", "show", "abl","abl50", "abl20"], default="rsample3", help="TBD")       
    parser.add_argument("--model", choices=ALL_MODELS + ['all'], default="resnet50", help="TBD")    
    parser.add_argument('--ext', action='store_true', default=False, help="Enable extended mode")
    parser.add_argument('--equant', action='store_true', default=False, help="Enable extended mode")    

    args = parser.parse_args()    
    return args

def print_hw():

    hostname = socket.gethostname()
    print("Hostname:", hostname)

    print("CPU Info:")
    print("="* 20)
    with open("/proc/cpuinfo", "r") as cf:
        print(cf.read())

    print("GPU Info:")
    print("="* 20)
    if torch.cuda.is_available():        
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            print(f"  CUDA Device ID: {i}")
    else:
        print("No GPU available.")

if __name__ == '__main__':
        
    print_hw()
    logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
    logging.info("start")    
    args = get_args()
    
    task_id = int(os.environ.get('SLURM_PROCID'))
    ntasks = int(os.environ.get('SLURM_NTASKS'))
                   
    logging.debug(args)
    logging.debug(f"pid: {os.getpid()}; task: {task_id}/{ntasks}")
    if args.dataset == "imagenet":
        isrc = ImagenetSource(selection_name=args.selection)
    elif args.dataset == "voc":
        isrc = VOCSource(selection_name=args.selection)
    else:
        assert False, "unexpected dataset"
    
    all_images_dict = isrc.get_all_images()
    all_images = sorted(list(all_images_dict.values()), key=lambda x:x.name)
    all_image_names = set(all_images_dict.keys())

    task_images = [img for idx, img in enumerate(all_images) if idx % ntasks == task_id]

    progress_path = os.path.join("progress", args.model, f"{args.action}_{args.sal}_{args.marker}")
    coord_images = Coord(all_images, progress_path)

    logging.info(f"images: {len(task_images)}/{len(all_images)}")

    try:
        if args.action == "list_images":
            for img in task_images:
                print(f"{img.name}")
        elif args.action == "summary":
            if args.model == "all":
                model_names = ALL_MODELS
            else:
                model_names = [args.model]
            for name in model_names:
                create_model_summary(name, extended=args.ext, equant=args.equant)
        elif args.action == "asummary":
            create_model_summary_all(args.model)
        elif args.action == "create_sals":        
            model_name = args.model
            sal_names = args.sal
            create_model_sals(model_name, sal_names, args.marker)
        elif args.action == "scores":
            if args.model == 'all':
                model_names = ALL_MODELS
            else:
                model_names = [args.model]
            for model_name in model_names:
                logging.info(f"##### scores {model_name} ######")
                create_model_scores(model_name, args.marker, extended=args.ext, subset=all_image_names, equant=args.equant)
        elif args.action == "all":
            for model_name in ALL_MODELS:
                logging.info(f"##### saliency {model_name} ######")
                create_model_sals(model_name, "all", args.marker)
            for model_name in ALL_MODELS:
                logging.info(f"##### scores {model_name} ######")            
                create_model_scores(model_name, args.marker)
            for model_name in ALL_MODELS:
                logging.info(f"##### scores (2) {model_name} ######")            
                create_model_scores(model_name, args.marker)
    except:
        logging.exception("error")
        raise
    finally:
        logging.info("done")

            


        
    