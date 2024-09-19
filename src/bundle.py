#!/bin/python
import sys, os, subprocess, glob


methods = [
    "PCPE_64_4000_ipw_0.1_0",
    "PCPE_64_2000_ipw_0.1_0",
    "PCPE_64_4000_ate_0.1_0",
    "PCPE_64_2000_ate_0.1_0",
    "PCPE_64_1000_ate_0",
    "PCPE_64_500_ate_0",
    "PCPE_64_100_ate_0",
    "PCPE_64_10_ate_0",
]

images = [
    "ILSVRC2012_val_00043161",
    "ILSVRC2012_val_00023950",
    "ILSVRC2012_val_00037825",
    "ILSVRC2012_val_00046566",
    "ILSVRC2012_val_00006117"
]

BASE_PATH = "/home/weziv5/work"
DEST_PATH = os.path.join(BASE_PATH, "report", "bundle")
IMG_PATH = os.path.join(BASE_PATH, "data", "imagenet", "validation")



def shell(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd)
    
os.makedirs(DEST_PATH, exist_ok=True)    
for image_name in images:    
    shell(["cp", os.path.join(IMG_PATH, f"{image_name}.JPEG"), DEST_PATH])

    ptrn = os.path.join(BASE_PATH, f"saliency.run/results/*/saliency/*/{image_name}")
    print("## ", ptrn)
    sal_paths = glob.glob(ptrn)
    
    print("### ", sal_paths)
    for path in sal_paths:
        parts = path.split("/")
        model_name, _, mthd, iname = parts[-4:]
        assert iname == image_name ##+ '.JPEG' 
        
        print("checking", path, model_name, mthd, image_name)
        if mthd not in methods:
            print("skipping")
            continue
        print("including")
        sal_name =f"SAL-{model_name}-{mthd}-{image_name}"
        print(sal_name)
        shell(["cp", path, os.path.join(DEST, sal_name)]) 

        
        



