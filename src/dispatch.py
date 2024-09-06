#!/bin/python


import argparse, logging, os
from dataset import ImagenetSource
from adaptors import CamSaliencyCreator, METHOD_CONV

from benchmark import *
from cpe import *

def create_cpe_sals(me, images, segsize=64):
    algo = IpwSalCreator(f"CPE_{segsize}", [500,1000,2000,4000], segsize=segsize, batch_size=32)
    logging.info("creating saliency maps")    
    create_saliency_data(me, algo, images, run_idx=0, exist_name="cpe", with_scores=False)

def create_cam_sals(me, images):
    algo = CamSaliencyCreator(list(METHOD_CONV.key()))
    create_saliency_data(me, algo, images, run_idx=0, exist_name="camsal", with_scores=False)

def get_args(): 
        
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images", "create_sals"], help="TBD")
    parser.add_argument("--sal", choices=["cpe","cam"], default="cpe", help="TBD")       
    parser.add_argument("--selection", choices=["dbl","selection0"], default="selection0", help="TBD")       
    parser.add_argument("--model", choices=["resnet18","resnet50"], default="resnet50", help="TBD")       

    args = parser.parse_args()    
    return args

if __name__ == '__main__':
        
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.DEBUG)
    logging.info("start")    
    args = get_args()
    
    task_id = int(os.environ.get('SLURM_PROCID'))
    ntasks = int(os.environ.get('SLURM_NTASKS'))
                   
    logging.debug(args)
    logging.debug(f"pid: {os.getpid()}; task: {task_id}/{ntasks}")
    isrc = ImagenetSource(selection_name=args.selection)
    
    all_images = sorted(list(isrc.get_all_images().values()), key=lambda x:x.name)
    task_images = [img for idx, img in enumerate(all_images) if idx % ntasks == task_id]

    logging.info(f"images: {len(task_images)}/{len(all_images)}")

    if args.action == "list_images":
        for img in task_images:
            print(f"{img.name}")
        
    else:
        me = ModelEnv(args.model)
        if args.action == "create_sals":
            if args.sal == "cpe":
                create_cpe_sals(me, task_images)
            elif args.action == "cam":
                create_cam_sals(me, task_images)

        
    