#!/bin/python
import sys, os, subprocess, glob

methods_ = [
    "PCPE_64_4000_ipw_0.1_0",
    "PCPE_64_2000_ipw_0.1_0",
    "PCPE_64_2000_ate_0",
    "PCPE_64_4000_ate_0",
    "pgc_GradCAM_0",
    "pgc_AblationCAM_0",
    "pgc_GradCAMPlusPlus_0",
    "pgc_LayerCAM_0",
    "pgc_FullGrad_0",
    "RISE_4000_7_0.5_0",
    "CexCnn_0.95_0",
    "CexCnn_0.995_0",
    "CexCnn_0.75_0",
    "GIG_100_0",
    "IG_100_0",
    "dimpl_dix_0",
    "dimpl_t-attr_0",
    "dimpl_gae_0"
]

__methods_ = [
    "PCPE_64_4000_ipw_0.1_0",
    "PCPE_64_2000_ipw_0.1_0",
    "PCPE_64_4000_ate_0.1_0",
    "PCPE_64_2000_ate_0.1_0",
    "PCPE_64_1000_ate_0",
    "PCPE_64_500_ate_0",
    "PCPE_64_100_ate_0",
    "PCPE_64_10_ate_0",
    "RISE_4000_7_0.5_0",
    "pgc_GradCAM_0",
    "pgc_AblationCAM_0",
    "pgc_FullGrad_0",
    "pgc_GradCAMPlusPlus_0",
    "dimpl_dix_0",
    "dimpl_t-attr_0",
    "dimpl_gae_0",
]


methods_ = [
    "PCPE_64_4000_ipw_0.1_0",
    "PCPE_64_2000_ipw_0.1_0",
    "PCPE_64_2000_ate_0",
    "PCPE_64_4000_ate_0",
    "pgc_GradCAM_0",
    "pgc_AblationCAM_0",
    "pgc_GradCAMPlusPlus_0",
    "pgc_LayerCAM_0",
    "pgc_FullGrad_0",
    "RISE_4000_7_0.5_0",
    "CexCnn_0.95_0",
    "CexCnn_0.995_0",
    "CexCnn_0.75_0",
    "GIG_100_0",
    "IG_100_0",
    "dimpl_dix_0",
    "dimpl_t-attr_0",
    "dimpl_gae_0"
]

methods = [
    "ABL_20_2000_ate_0",
    "ABL_32_2000_ate_0",
    "ABL_48_2000_ate_0",
    "ABL_64_2000_ate_0",
    "ABL_80_2000_ate_0",
    "ABL_20_2000_ipw_0.1_0",
    "ABL_32_2000_ipw_0.1_0",
    "ABL_48_2000_ipw_0.1_0",
    "ABL_64_2000_ipw_0.1_0",
    "ABL_80_2000_ipw_0.1_0",

]


images = "ILSVRC2012_val_00033991.JPEG,ILSVRC2012_val_00006548.JPEG,ILSVRC2012_val_00033042.JPEG,ILSVRC2012_val_00047485.JPEG,ILSVRC2012_val_00003509.JPEG,ILSVRC2012_val_00018756.JPEG,ILSVRC2012_val_00026342.JPEG,ILSVRC2012_val_00010128.JPEG,ILSVRC2012_val_00034719.JPEG,ILSVRC2012_val_00032452.JPEG,ILSVRC2012_val_00031219.JPEG,ILSVRC2012_val_00006339.JPEG,ILSVRC2012_val_00010034.JPEG,ILSVRC2012_val_00011566.JPEG,ILSVRC2012_val_00027999.JPEG,ILSVRC2012_val_00025894.JPEG,ILSVRC2012_val_00005080.JPEG,ILSVRC2012_val_00001651.JPEG,ILSVRC2012_val_00040839.JPEG,ILSVRC2012_val_00000438.JPEG,ILSVRC2012_val_00038147.JPEG,ILSVRC2012_val_00019064.JPEG,ILSVRC2012_val_00000033.JPEG,ILSVRC2012_val_00034284.JPEG,ILSVRC2012_val_00037768.JPEG,ILSVRC2012_val_00033897.JPEG,ILSVRC2012_val_00041069.JPEG,ILSVRC2012_val_00037572.JPEG,ILSVRC2012_val_00019059.JPEG,ILSVRC2012_val_00003649.JPEG,ILSVRC2012_val_00035784.JPEG,ILSVRC2012_val_00000706.JPEG,ILSVRC2012_val_00009150.JPEG,ILSVRC2012_val_00041370.JPEG,ILSVRC2012_val_00005759.JPEG,ILSVRC2012_val_00000457.JPEG,ILSVRC2012_val_00004999.JPEG,ILSVRC2012_val_00015822.JPEG,ILSVRC2012_val_00009976.JPEG,ILSVRC2012_val_00004910.JPEG,ILSVRC2012_val_00036044.JPEG,ILSVRC2012_val_00006873.JPEG,ILSVRC2012_val_00011502.JPEG,ILSVRC2012_val_00013324.JPEG,ILSVRC2012_val_00036589.JPEG,ILSVRC2012_val_00008888.JPEG,ILSVRC2012_val_00025299.JPEG,ILSVRC2012_val_00013825.JPEG,ILSVRC2012_val_00019444.JPEG,ILSVRC2012_val_00014703.JPEG,ILSVRC2012_val_00022169.JPEG,ILSVRC2012_val_00023224.JPEG,ILSVRC2012_val_00031079.JPEG,ILSVRC2012_val_00040625.JPEG,ILSVRC2012_val_00049274.JPEG,ILSVRC2012_val_00023955.JPEG,ILSVRC2012_val_00042253.JPEG,ILSVRC2012_val_00032888.JPEG,ILSVRC2012_val_00028636.JPEG,ILSVRC2012_val_00024382.JPEG,ILSVRC2012_val_00028989.JPEG,ILSVRC2012_val_00012271.JPEG,ILSVRC2012_val_00008443.JPEG,ILSVRC2012_val_00029072.JPEG,ILSVRC2012_val_00023554.JPEG,ILSVRC2012_val_00004235.JPEG,ILSVRC2012_val_00024097.JPEG,ILSVRC2012_val_00016645.JPEG,ILSVRC2012_val_00007799.JPEG,ILSVRC2012_val_00046068.JPEG,ILSVRC2012_val_00001280.JPEG,ILSVRC2012_val_00033224.JPEG,ILSVRC2012_val_00044790.JPEG,ILSVRC2012_val_00002472.JPEG,ILSVRC2012_val_00049174.JPEG,ILSVRC2012_val_00013703.JPEG,ILSVRC2012_val_00034239.JPEG,ILSVRC2012_val_00018040.JPEG,ILSVRC2012_val_00023223.JPEG,ILSVRC2012_val_00041515.JPEG,ILSVRC2012_val_00012831.JPEG,ILSVRC2012_val_00010609.JPEG,ILSVRC2012_val_00012316.JPEG,ILSVRC2012_val_00040337.JPEG,ILSVRC2012_val_00035693.JPEG,ILSVRC2012_val_00005256.JPEG,ILSVRC2012_val_00049939.JPEG,ILSVRC2012_val_00021850.JPEG,ILSVRC2012_val_00045127.JPEG,ILSVRC2012_val_00044117.JPEG,ILSVRC2012_val_00026849.JPEG,ILSVRC2012_val_00002090.JPEG,ILSVRC2012_val_00046123.JPEG,ILSVRC2012_val_00034321.JPEG,ILSVRC2012_val_00046232.JPEG,ILSVRC2012_val_00023960.JPEG,ILSVRC2012_val_00043229.JPEG,ILSVRC2012_val_00032854.JPEG,ILSVRC2012_val_00008780.JPEG,ILSVRC2012_val_00037963.JPEG"
images = "ILSVRC2012_val_00014113.JPEG,ILSVRC2012_val_00022969.JPEG,ILSVRC2012_val_00020979.JPEG,ILSVRC2012_val_00000547.JPEG,ILSVRC2012_val_00017942.JPEG,ILSVRC2012_val_00023857.JPEG,ILSVRC2012_val_00022326.JPEG,ILSVRC2012_val_00009169.JPEG,ILSVRC2012_val_00015849.JPEG,ILSVRC2012_val_00012838.JPEG"
images = [x[0:x.find(".")] for x in images.split(",")]
images = images

BASE_PATH = "/home/weziv5/work"
DEST_PATH = os.path.join(BASE_PATH, "report", "bundle")
IMG_PATH = os.path.join(BASE_PATH, "data", "imagenet", "validation")



def shell(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd)
    
os.makedirs(DEST_PATH, exist_ok=True)    
for idx, image_name in enumerate(images):    
    print(f"========== {idx} {image_name} ===========")

    ptrn = os.path.join(BASE_PATH, f"saliency.abl/results/*/saliency/*/{image_name}")
    print("## ", ptrn)
    sal_paths = glob.glob(ptrn)
    
    if not sal_paths:
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

        
        



