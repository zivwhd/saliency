#!/bin/python
import sys, os, subprocess, glob


methods = [
    "PCPE_64_4000_ipw_0.1_0"
    "PCPE_64_10_ate_0"
]

images = [
    "ILSVRC2012_val_0000719"
]

BASE_PATH = "/home/weziv5/work"
DEST_PATH = os.path.join(BASE_PATH, "report", "bundle")
IMG_PATH = os.path.join(BASE_PATH, "data", "imagenet", "validation")



def shell(cmd):
    print(">> {cmd}")
    subprocess.run(cmd)
    
os.makedirs(DEST_PATH, exist_ok=True)    
for image_name in images:    
    shell("cp", os.path.join(IMG_PATH, f"{image_name}.JPEG"), DEST_PATH)

    sal_paths = glob.glob(os.path.join(BASE_PATH, f"saliency.run/results/*/saliency/*/{image_name}"))

    sal_paths = [x for x in sal_path if x.split("/")[-2] in methods]

    for path in sal_paths:
        parts = path.split("/")
        model_name, _, mthd, iname = parts[-4:]
        assert iname == image_name
        
        if mthd not in methods:
            print("skipping", path)
            continue
        print("including", path, model_name, mthd, image_name)
        sal_name =f"SAL-{model_name}-{mthd}-{image_name}"
        print(sal_name)

        
        



