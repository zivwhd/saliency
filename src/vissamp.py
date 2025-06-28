from intutils import *
from dataset import *
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import glob,os,logging,json,argparse

matplotlib.use('Agg')

logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)


def get_args(): 
    VIT_MODELS = ["vit_small_patch16_224","vit_base_patch16_224","vit_base_patch16_224.mae"]
    CNN_MODELS = ["resnet50","vgg16", "convnext_base", "densenet201","densenet201NT","densenet201RT","resnet50NT", "resnet50RT", "vgg16NT", "vgg16RT"] ## "resnet18"
    ALL_MODELS = CNN_MODELS + VIT_MODELS    
    parser = argparse.ArgumentParser(description="dispatcher")
    parser.add_argument("--selection")       
    parser.add_argument("--model", choices=ALL_MODELS, default="resnet50", help="TBD")    

    args = parser.parse_args()    
    return args

args = get_args()

selection = args.selection
model_name = args.model
isrc = ImagenetSource(selection_name=selection)


if 'vit' not in model_name:
    methods = [
        ('SLOC', 'AutoComp_1000_32_101_msk1.0_tv0.1_mgn0.01_0'),
        ('AC', 'pgc_AblationCAM_0'),
        ('DIX', 'DixCnn_0'),
        ('EP','MPert_300_o1.0_tv2_2_l0.2_0'),
        ('GC', 'pgc_GradCAM_0'),
        ('LTX','sLTX_50_5_5e-05_1.0_0.5_0'),
        ( 'RISE','RISE_4000_7_0.5_0')
]
else:
    methods = [
        ('SLOC', 'AutoComp_1000_32_101_msk1.0_tv0.1_mgn0.01_0'),        
        ('DIX', 'Dimpl_dix_0'),
        ('EP','MPert_300_o1.0_tv2_2_l0.2_0'),
        ('GAE', 'Dimpl_gae_0'),        
        ('LTX','sLTX_50_5_5e-05_1.0_0.5_0'),
        ( 'RISE','RISE_4000_7_0.5_0'),    
        ('T-Attr', 'Dimpl_t-attr_0'),
    ]

methods =[('SLOCpos','AutoCompPos_1000_32_101_msk1.0_tv0.1_p1_0')]
TARGET_NAMES = json.load(open(os.path.join('dataset','imagenet_class_index.json')))


figsize=(10,10)
fontsize=7

all_images_dict = isrc.get_all_images()
all_images = sorted(list(all_images_dict.values()), key=lambda x:x.name)
all_image_names = set(all_images_dict.keys())

for imgidx, image_info in enumerate(all_images):
    logging.info(f"[{imgidx}] image: {image_info}")
    image_path = image_info.path
    image_name = image_info.name
    targetidx = image_info.target
    ## {"0": ["n01440764", "tench"]
    target_name = TARGET_NAMES[str(targetidx)][1]
    logging.info(f"target: {targetidx} {target_name}")
    img=Image.open(image_path)
    img=img.resize((224,224))  

    if figsize:
        plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.tight_layout(pad=0)


    
    idx = 1
    plt.subplot(1, len(methods)+1, idx)             
    idx += 1
    show_single_sal(img, None, None)   
    plt.figtext(0.98, 0.5, target_name, va='center', ha='left', rotation='vertical')



    for method_name, variant in methods:
        logging.info(f"method: {method_name} - {variant}")
        result_path = os.path.join("results", model_name, "saliency", variant, image_info.name)
        logging.info(f"loading {result_path}")
        assert os.path.isfile(result_path)
        sal = torch.load(result_path).float()
        logging.info(f"loaded {sal.shape}")
        plt.subplot(1, len(methods)+1, idx)         
        idx += 1
        show_single_sal(img, None, None)
        show_single_sal(img, {method_name : sal}, method_name, alpha=0.6, mag=True)
        plt.title(method_name, fontsize=fontsize)
    
    
    save_path = f"visual/{model_name}/{image_name}.png"
    logging.info(f"saving: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=1200, bbox_inches='tight', transparent=False, pad_inches=0)    


