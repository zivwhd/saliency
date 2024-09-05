#!/bin/python
print("########### python ############")

import argparse, logging, os
from dataset import ImagenetSource


def get_args(): 
    
    

    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images"], help="TBD")
    parser.add_argument("--selection", choices=["dbl","selection0"], default="selection0", help="TBD")       
    #parser.add_argument("--env", type=str, help="TBD")

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


    if args.action == "list_images":
        for img in task_images:
            print(f"{img.name}")
        
    