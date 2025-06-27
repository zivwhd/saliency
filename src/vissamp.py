from intutils import *
from dataset import *
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import glob,os,logging

matplotlib.use('Agg')

logging.basicConfig(format='[%(asctime)-15s  %(filename)s:%(lineno)d - %(process)d] %(message)s', level=logging.DEBUG)

selection = "vis"
isrc = ImagenetSource(selection_name=selection)
image_name = "ILSVRC2012_val_00032607"
model_name = "resnet50"

methods = [
    ('LSC', 'AutoComp_1000_32_101_msk1.0_tv0.1_mgn0.01_0'),
    ('DIX', 'DixCnn_0'),
    ('GC', 'pgc_GradCAM_0')
]


figsize=(10,10)
fontsize=7

all_images_dict = isrc.get_all_images()
all_images = sorted(list(all_images_dict.values()), key=lambda x:x.name)
all_image_names = set(all_images_dict.keys())

for image_info in all_images:
    logging.info(f"image: {image_info}")
    image_path = image_info.path
    targetidx = image_info.target
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
        show_single_sal(img, sal, method_name) #, alpha=alpha, mag=mag)
        plt.title(method_name, fontsize=fontsize)
    
    plt.savefig("visual/{model_name}/{image_name}.png") 
    save_path = "visual/{model_name}/{image_name}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=1200, bbox_inches='tight', transparent=False, pad_inches=0)

