import sys, os, subprocess


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
    
os.makedirs(dest, exist_ok=True)    
for image_name in images:    
    shell("cp", os.path.join(IMG_PATH, f"{image_name}.JPEG"), DEST_PATH)

    sal_paths = glob.glob(os.path.join(BASE_PATH, f"saliency.run/results/*/saliency/*/{image_name}"))

    sal_paths = [x for x in sal_path if x.split("/")[-2] in methods]

    for path in sal_paths:
        mthd = x.split("/")[-2]
        if mthd not in methods:
            print("skipping", path)
            continue
        print("including", path)
        

    for method in methods:

        sal_path = os.path.join(BASE_PATH, "saliency.run/results")
        
        



