#!/bin/python

print("## dispatch.py ")

import argparse, logging, os, re
from dataset import ImagenetSource, Coord
from adaptors import CaptumCamSaliencyCreator, CamSaliencyCreator, METHOD_CONV
from vit_adaptors import AttrVitSaliencyCreator, DimplVitSaliencyCreator
from RISE import RiseSaliencyCreator
from benchmark import *
from cpe import *


def get_cpe_sal_creator(segsize=64):
    return IpwSalCreator(f"CPE_{segsize}", [10, 100, 250, 500,1000,2000,4000], segsize=segsize, batch_size=32)

def get_pcpe_sal_creator(segsize=64):
    return IpwSalCreator(f"PCPE_{segsize}", [10, 100, 250, 500, 1000, 2000,4000], segsize=segsize, with_softmax=True, batch_size=32)

def get_rcpe_sal_creator(segsize=64):
    return IpwSalCreator(f"RCPE_{segsize}", [500,1000,2000,4000], segsize=segsize, batch_size=32, ipwg=RelIpwGen)

def get_cam_sal_creator():
    return CamSaliencyCreator()

def get_tattr_sal_creator():
    return AttrVitSaliencyCreator()

def get_dimpl_sal_creator():
    return DimplVitSaliencyCreator()

def get_rise_sal_creator():
    return RiseSaliencyCreator()

def get_captum_sal_creator():
    return CaptumCamSaliencyCreator()


ALL_CNN_CREATORS = ["pcpe", "cam", "captum", "rise"]
ALL_VIT_CREATORS = ["pcpe", "dimpl", "tattr", "rise"]

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

def create_model_scores(model_name, marker="c1"):
    me = ModelEnv(model_name)            
    result_paths = get_all_results(model_name)
    logging.info(f"found {len(result_paths)} saliency maps")
    progress_path = os.path.join("progress", args.model, f"scores_any_{marker}")
    result_prog = Coord(result_paths, progress_path, getname=get_score_name)            
    create_scores(me, result_prog, all_images_dict, update=True)

def create_model_summary(model_name):
    base_csv_path = os.path.join("results", model_name)
    df = load_scores_df(model_name)
    df.to_csv(f'{base_csv_path}/results.csv', index=False)
    smry = summarize_scores_df(df)
    smry.to_csv(f'{base_csv_path}/summary.csv', index=False)

def get_creators():
    ptrn = re.compile("get_(.*)_sal_creator")
    return [match.group(1) for match in [ptrn.match(vr) for vr in globals()] if match is not None]

VIT_MODELS = ["vit_small_patch16_224","vit_base_patch16_224"]
CNN_MODELS = ["resnet50","vgg16"] ## "resnet18"
ALL_MODELS = CNN_MODELS + VIT_MODELS

def get_args(): 
    creators = get_creators() + ['any','all']
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images", "create_sals", "scores", "summary", "all"], help="TBD")
    parser.add_argument("--sal", choices=creators, default="cpe", help="TBD")
    parser.add_argument("--marker", default="m", help="TBD")       
    parser.add_argument("--selection", choices=["rsample3", "rsample100", "rsample1000"], default="rsample3", help="TBD")       
    parser.add_argument("--model", choices=CNN_MODELS, default="resnet50", help="TBD")    

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
        create_model_summary(args.model)
    elif args.action == "create_sals":        
        model_name = args.model
        sal_names = args.sal
        create_model_sals(model_name, sal_names, args.marker)
    elif args.action == "scores":
        create_model_scores(args.model, args.marker)
    elif args.action == "all":
        for model_name in ALL_MODELS:
            logging.info("##### {model_name} ######")
            create_model_sals(model_name, "all", args.marker)
            create_model_scores(model_name, args.marker)
            create_model_summary(model_name)
            


        
    