#!/bin/python
print("########### python ############")

import argparse, logging, os
from dataset import ImagenetSource


def get_args(): 
    
    task_id = os.environ.get('SLURM_PROCID')

    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images"], help="TBD")
    parser.add_argument("--selection", choices=["selection0"], help="TBD")       
    #parser.add_argument("--env", type=str, help="TBD")

    args = parser.parse_args()    
    return args

if __name__ == '__main__':
        
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.DEBUG)
    logging.info("start")    
    args = get_args()

    logging.debug(args)
    
    isrc = ImagenetSource(args.selection)
    images = isrc.get_all_images()

    if args.action == "selection":
        pass
    