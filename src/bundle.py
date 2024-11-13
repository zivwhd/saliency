#!/bin/python
import sys, os, subprocess, glob

methods = [
"MCompZr_1000_16_200:200_n_msk1.0_tv0.1_mdl0.05_0",
"RISE_4000_7_0.5_0",
"MCompZr_1000_16_200_n_msk1.0_tv0.1_0",
"sLTX_50_5_5e-05_1.0_0.5_0",
"sLTX_50_5_5e-05_cp1_1.0_0.5_0",
"IEMPert_300_tv0.2_1_l0.005_0",
"Dimpl_t-attr_0",
"Dimpl_dix_0",
"Dimpl_gae_0"]

methods = [
    "MDL_500_40_300:300b_msk1.0_tv0.1_mgn0.01_mdl0.01_0",
    "MDL_500_40_300:300b_msk1.0_tv0.1_mgn0.01_mdl0.05_0",
    "MDL_500_40_300:300b_msk1.0_tv0.1_mgn0.01_mdl0.1_0",
    "MDL_500_40_300:300b_msk1.0_tv0.1_mgn0.01_mdl0.2_0"
]

methods =[ 
    "ExtPertM_0",  "ExtPertS_0",  "ExtPert_0.05_0",  "ExtPert_0.1_0",  
    "ExtPert_0.2_0",  "ExtPert_0.4_0",  "ExtPert_0.6_0",  "ExtPert_0.8_0"]


BASE_PATH = "/home/weziv5/work"
SAMP_PATH = os.path.join(BASE_PATH, "data", "imagenet", "vis.smp")

with open(SAMP_PATH, "rt") as sf:
    images = sf.readlines()

images = [x.strip().replace(".JPEG","") for x in images]
images = images[0:100]

DEST_PATH = os.path.join(BASE_PATH, "report", "bundle2")
IMG_PATH = os.path.join(BASE_PATH, "data", "imagenet", "validation")



def shell(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd)
    
os.makedirs(DEST_PATH, exist_ok=True)    
for idx, image_name in enumerate(images):    
    print(f"========== {idx} {image_name} ===========")

    ptrn = os.path.join(BASE_PATH, f"saliency.run/results/*/saliency/*/{image_name}")
    print(f"##  '{ptrn}'")
    sal_paths = glob.glob(ptrn)
    
    if not sal_paths:
        print("- skipping")
        continue

    shell(["cp", os.path.join(IMG_PATH, f"{image_name}.JPEG"), DEST_PATH])

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
        sal_name =f"SAL-{model_name}-{mthd.replace('-','')}-{image_name}"
        print(sal_name)
        shell(["cp", path, os.path.join(DEST_PATH, sal_name)]) 

        
        



