#!/bin/python
import argparse, logging

from dataset import ImagenetSource


def get_args():    
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--action", choices=["list_images"], help="TBD")
    parser.add_argument("--selection", choices=["selection0"], help="TBD")       
    #parser.add_argument("--env", type=str, help="TBD")

    args = parser.parse_args()    
    return args

if __name__ == '__main__':
    
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.DEBUG)
    
    args = get_args()
    logging.debug(args)

    isrc = ImagenetSource(args.selection)
    images = isrc.get_all_images()

    if args.action == "selection":
        pass
    