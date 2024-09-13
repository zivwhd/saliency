#!/bin/python

print("## dispatch.py ")

import argparse, logging, os
from dataset import ImagenetSource, Coord
from adaptors import CamSaliencyCreator, METHOD_CONV
from vit_adaptors import AttrVitSaliencyCreator, DimplVitSaliencyCreator
from benchmark import *
from cpe import *

def create_cpe_sals(me, images, segsize=64):
    logging.info("create_cpe_sals")
    algo = IpwSalCreator(f"CPE_{segsize}", [500,1000,2000,4000], segsize=segsize, batch_size=32)
    logging.info("creating saliency maps")    
    create_saliency_data(me, algo, images, run_idx=0)

def create_rcpe_sals(me, images, segsize=64):
    logging.info("create_rcpe_sals")
    algo = IpwSalCreator(f"RCPE_{segsize}", [500,1000,2000,4000], segsize=segsize, batch_size=32, ipwg=RelIpwGen)
    logging.info("creating saliency maps")    
    create_saliency_data(me, algo, images, run_idx=0)

def create_cam_sals(me, images):
    logging.info("create_cam_sals")
    algo = CamSaliencyCreator(list(METHOD_CONV.keys()))
    create_saliency_data(me, algo, images, run_idx=0)

def create_tattr_sals(me, images):
    logging.info("create_tattr_sals")
    algo = AttrVitSaliencyCreator()
    create_saliency_data(me, algo, images, run_idx=0)

def create_dimpl_sals(me, images):
    logging.info("create_dimpl_sals")
    algo = DimplVitSaliencyCreator()
    create_saliency_data(me, algo, images, run_idx=0)

def get_args(): 
        
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images", "create_sals", "scores", "summary"], help="TBD")
    parser.add_argument("--sal", choices=["cpe","cam", "rcpe", "tattr", "dimpl", "any"], default="cpe", help="TBD")
    parser.add_argument("--marker", default="m", help="TBD")       
    parser.add_argument("--selection", choices=["rsample3", "rsample100", "rsample1000"], default="rsample3", help="TBD")       
    parser.add_argument("--model", choices=["resnet18","resnet50","vit_small_patch16_224"], default="resnet50", help="TBD")    

    args = parser.parse_args()    
    return args

if __name__ == '__main__':
        
    logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
    logging.info("start")    
    args = get_args()
    
    task_id = int(os.environ.get('SLURM_PROCID'))
    ntasks = int(os.environ.get('SLURM_NTASKS'))
                   
    logging.debug(args)
    logging.debug(f"pid: {os.getpid()}; task: {task_id}/{ntasks}")
    isrc = ImagenetSource(selection_name=args.selection)
    
    all_images_dict = isrc.get_all_images()
    all_images = sorted(list(all_images_dict.values()), key=lambda x:x.name)
    task_images = [img for idx, img in enumerate(all_images) if idx % ntasks == task_id]

    progress_path = os.path.join("progress", args.model, f"{args.action}_{args.sal}_{args.marker}")
    coord_images = Coord(all_images, progress_path)

    logging.info(f"images: {len(task_images)}/{len(all_images)}")

    if args.action == "list_images":
        for img in task_images:
            print(f"{img.name}")
    elif args.action == "summary":
        base_csv_path = os.path.join("results", args.model)
        df = load_scores_df(args.model)
        df.to_csv(f'{base_csv_path}/results.csv', index=False)

        smry = summarize_scores_df(df)
        smry.to_csv(f'{base_csv_path}/summary.csv', index=False)

    else:
        me = ModelEnv(args.model)
        if args.action == "create_sals":
            if args.sal == "cpe":
                create_cpe_sals(me, coord_images)
            if args.sal == "rcpe":
                create_rcpe_sals(me, coord_images)
            elif args.sal == "cam":
                create_cam_sals(me, coord_images)
            elif args.sal == "tattr":
                create_tattr_sals(me, coord_images)

            elif args.sal == "any":
                assert False, "unexpected sal"
        elif args.action == "scores":            
            result_paths = get_all_results(args.model)
            logging.info(f"found {len(result_paths)} saliency maps")
            result_prog = Coord(result_paths, progress_path, getname=get_score_name)            
            create_scores(me, result_prog, all_images_dict, update=True)
            


        
    