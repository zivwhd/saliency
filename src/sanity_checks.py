import torch
import numpy as np
import pandas as pd
import glob
import os, sys, logging

logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)


import torch

def spearman_rank_correlation(x, y):
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    # Rank the elements
    x_rank = torch.argsort(torch.argsort(x))
    y_rank = torch.argsort(torch.argsort(y))
    
    # Compute the rank differences
    d = x_rank.float() - y_rank.float()
    d_squared = d ** 2
    
    # Number of elements
    n = x.shape[0]
    
    # Compute Spearman's rank correlation
    numerator = 6 * d_squared.sum()
    denominator = n * (n ** 2 - 1)
    rho = 1 - (numerator / denominator)
    
    return rho.item()



BASE_PATH = "resultsA/densenet201/saliency/"
def stats():
    map_location = torch.device('cpu')
    ptrn = os.path.join(BASE_PATH, "Base_0", "*")
    logging.info(f">> {ptrn}")
    base_paths = glob.glob(ptrn)
    logging.info(f"images: {base_paths[0:5]}")
    logging.info(f"num images: {len(base_paths)}")
    rows = []
    for path in base_paths[0:5]:
        logging.info(f">> {path}")
        image_name = os.path.basename(path)
        base = torch.load(path, map_location=map_location)

        for rtype in ["Rnd","Csc"]:
            for lidx in range(1, 33):
                other_path = os.path.join(BASE_PATH, f"{rtype}_{lidx}_0", image_name)
                other = torch.load(other_path, map_location=map_location)
                scor = spearman_rank_correlation(base.flatten(), other.flatten())
                rows.append(dict(rtype=rtype, image=image_name, scor=scor, idx=lidx))

        for mdl in ["NT","RT"]:
            other_path = "resultsA/desnenet201{msdl}/saliency/MWCompZr_500_40_300b_msk1.0_tv0.1_mgn0.01_0/{image_name}"
            other = torch.load(other_path, map_location=map_location)
            scor = spearman_rank_correlation(base.flatten(), other.flatten())
            rows.append(dict(rtype=mdl, image=image_name, scor=scor, idx=0))
            
        logging.info(f"total rows: {len(rows)}")
    df = pd.DataFrame(rows)
    df.to_csv("results/sanity.csv")

stats()

                           


