#!/bin/python

print("## dispatch.py ")

import argparse, logging, os, re,sys
from dataset import ImagenetSource, VOCSource, Coord
import torch

from benchmark import *
from cpe import *
from lcpe import CompExpCreator, MultiCompExpCreator, AutoCompExpCreator, MProbCompExpCreator, ZeroBaseline, RandBaseline, BlurBaseline, MulCompExpCreator, ProbRangeCompExpCreator, SegSlocExpCreator, RngSegSlocExpCreator


def get_args(): 
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--selection", default="rsample3", help="TBD")       
    parser.add_argument("--marker", default="BBB", help="TBD")       
    parser.add_argument("--model", default="resnet50", help="TBD")
    parser.add_argument("--mag", type=float, default=0, help="TBD")
    args = parser.parse_args()    
    return args

if __name__ == '__main__':
        
    
    logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
    logging.info("start")    
    args = get_args()

    isrc = ImagenetSource(selection_name=args.selection)
    all_images_dict = isrc.get_all_images()
    all_images = sorted(list(all_images_dict.values()), key=lambda x:x.name)
    all_image_names = set(all_images_dict.keys())

    progress_path = os.path.join("progress", args.model, f"ateest_{args.model}_{args.marker}")
    coord_images = Coord(all_images, progress_path)

    me = ModelEnv(args.model)    

    for itr, info in enumerate(coord_images):    
        
        image_name = info.name
        image_path = info.path 
        target = info.target
        magnitude = args.mag
        pidx = image_name.find(".")
        if pidx > 0:
            image_name = image_name[0:pidx]

        try:
            img, inp = me.get_image_ext(image_path)
        except:
            logging.exception("Failed getting image")
            logging.info("Skipping")
            continue


        segsize = 56
        logits = me.model(inp).cpu()
        topidx = int(torch.argmax(logits))        
        prob = float(torch.softmax(logits, dim=1)[0,topidx])
        logging.info(f"ext {itr} {image_path} {image_name} {topidx} {info.desc} : {prob}")
        
        algo = CompExpCreator(nmasks=1000, segsize=segsize, pprob=0.5, 
                              epochs=None, c_tv=100, c_magnitude=magnitude )
        data = algo.generate_data(me, inp, topidx)    
        res = algo(me, inp,topidx,data)
        sal = list(res.values())[0].squeeze().cpu()

        fx, fy = random.randrange(0, segsize), random.randrange(0, segsize)
        #fx, fy = 0, 0 
        sq = SqMaskGen(segsize, (224,224), efactor=4, fcrop=(fx,fy))
        sqalgo = CompExpCreator(nmasks=2000, segsize=42, pprob=0.5, epochs=None, c_tv=100, c_magnitude=0, mgen=sq)
        sqdata = sqalgo.generate_data(me, inp, topidx)

        all_pred = sqdata.all_pred / inp.numel()
        Y1 = ((sqdata.all_masks > 0.5)*all_pred.unsqueeze(1).unsqueeze(1)).sum(dim=0) / (sqdata.all_masks > 0.5).sum(dim=0)
        Y0 = ((sqdata.all_masks < 0.5)*all_pred.unsqueeze(1).unsqueeze(1)).sum(dim=0) / (sqdata.all_masks < 0.5).sum(dim=0)
        ATE = (Y1-Y0).cpu()
        ATE.shape
        ATE

        ###
        ate_list = []
        sal_list = []
        fac = inp.numel()
        seg = sq.segments[sq.fcrop[0]:(sq.fcrop[0]+224), sq.fcrop[1]:(sq.fcrop[1]+224)]
        sids = seg.unique().tolist()
        for idx,id in enumerate(sids):
            sel = (seg == id)
            ate_val = ATE[sel].mean() 
            sal_val = sal[sel].sum() / (4*3)
            #print(ate_val, sal_val)
            ate_list.append(ate_val)
            sal_list.append(sal_val)
            #title,model,marker,magnitude,image,topidx,prob,fx,fy,idx,id,ate_val,sal_val
            print(f"ATEEST,{args.model},{args.marker},{args.mag},{segsize},{image_name},{topidx},{prob},{fx},{fy},{idx},{id},{ate_val},{sal_val}")

        sys.stdout.flush()
        #sal = 4 * (sel * sal).sum()
    #print(aaa, bbb)

        #esal = torch.Tensor(sal_list)
        #eate = torch.Tensor(ate_list)