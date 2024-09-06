#!/bin/python

print("## dispatch.py ")

import argparse, logging, os
from dataset import ImagenetSource
from adaptors import CamSaliencyCreator, METHOD_CONV

from benchmark import *
from cpe import *

def create_cpe_sals(me, images, segsize=64):
    logging.info("create_cpe_sals")
    algo = IpwSalCreator(f"CPE_{segsize}", [500,1000,2000,4000], segsize=segsize, batch_size=32)
    logging.info("creating saliency maps")    
    create_saliency_data(me, algo, images, run_idx=0, exist_name="cpe", with_scores=False)

def create_cam_sals(me, images):
    logging.info("create_cam_sals")
    algo = CamSaliencyCreator(list(METHOD_CONV.keys()))
    create_saliency_data(me, algo, images, run_idx=0, exist_name="camsal", with_scores=False)

def get_args(): 
        
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images", "create_sals", "scores"], help="TBD")
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
    task_image_dict = {info.name : info for info in task_images }

    logging.info(f"images: {len(task_images)}/{len(all_images)}")

    if args.action == "list_images":
        for img in task_images:
            print(f"{img.name}")
        
    else:
        me = ModelEnv(args.model)
        if args.action == "create_sals":
            if args.sal == "cpe":
                create_cpe_sals(me, task_images)
            elif args.sal == "cam":
                create_cam_sals(me, task_images)
        elif args.action == "scores":            
            result_paths = get_all_results()
            logging.info(f"found {len(result_paths)} saliency maps")
            create_scores(me, task_image_dict, result_paths, update=True)
            


        
    