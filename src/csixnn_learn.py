import os, logging, argparse, sys
from dataset import Coord, ImagenetSource
from csixnn import *
from benchmark import ModelEnv
import functools

MODELS = ['resnet50', 'vgg16','convnext_base']

def get_args():     
    parser = argparse.ArgumentParser(description="learn ixnn causal graph")
    parser.add_argument("--selection", choices=["remaining"], default="remaining", help="TBD")       
    parser.add_argument("--model", choices=MODELS + ['all'], default="resnet50", help="TBD")    

    args = parser.parse_args()    
    return args


if __name__ == '__main__':
        
    logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)
    logging.info("start - learning ixnn - causal graph")
    print("IXNN - Learn")
    args = get_args()   
    logging.debug(args)
    isrc = ImagenetSource(selection_name=args.selection)
    me = ModelEnv(args.model)
    marker = "c1"
    progress_path = os.path.join("progress", args.model, f"cp_{marker}")
    target_list = list(range(1000))
    prog = Coord(target_list, progress_path, getname=str)

    for target in prog:
        if os.path.exists(get_cp_path(BASE_PATH, me.arch, target)):
            logging.info(f"overriding {target}")
        #    continue
        generate_causal_path(me, target, isrc, device='cuda')

