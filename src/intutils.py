import numpy as np
from matplotlib import pyplot as plt
import torch
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm,Normalize

def show_image(path):
    img=Image.open(path)
    img=img.resize((224,224))    
    plt.title('input image', fontsize=18)
    plt.imshow(img)


def show_sal_scores(img, scores_dict, sals_dict):
    plt.figure(figsize=(10, 3 * len(scores_dict)))

    idx = 1
    for name, scores in scores_dict.items():
                
        topidx = 0
        plt.subplot(len(scores_dict), 4, idx)
        isal = sals_dict[name][0].cpu()
        
        plt.title(name)
        plt.imshow(isal, cmap='coolwarm')#cmap='RdBu')
        plt.subplot(len(scores_dict), 4, idx+1)
        plt.imshow(img)    
        plt.imshow(isal, cmap='coolwarm', alpha=0.5)  # Set alpha for transparency
        #plt.axis('off')  # Hide the axis
        #plt.show()    
        plt.subplot(len(scores_dict), 4, idx+2)
        plot_scores("", "deletion", scores["del"])
        plt.subplot(len(scores_dict), 4, idx+3)
        plot_scores("", "insertion", scores["ins"])
        idx +=4
        
    plt.subplots_adjust(hspace=0.5)   
    plt.show()


def toquant(sal):
    shape = sal.shape
    sal = sal.flatten()
    sorted_indices = torch.argsort(sal)
    # Get the ranks of the original tensor elements
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(sal))
    # Normalize the ranks to get quantiles (between 0 and 1)    
    quantiles = ranks.float() / (sal.numel() - 1)
    return quantiles.reshape(shape)

def showsal(sal, img, caption="", quantile=0, mag=True, alpha=0.6, with_mask=True, with_pos=False,  save_path=None):
    #stdsal = np.array( ((sal - sal.min()) / (sal.max()-sal.min())).unsqueeze(-1)) 
    #stdsal = (stdsal > 0.7)
    slots = 3 + with_mask + with_pos
    osal = sal
    mask = (sal - sal.min()) / (sal.max()-sal.min())
    if mag:
        sal = torch.max(torch.min(sal, torch.quantile(sal,0.99)), torch.quantile(sal,0.01))
    plt.subplot(1, slots, 3)
    plt.title(caption)
    plt.imshow(sal, cmap='jet')#cmap='RdBu')
    plt.xticks([])  
    plt.yticks([])
    plt.subplot(1, slots, 2)
    plt.imshow(img)    
    plt.imshow(sal, cmap='jet', alpha=alpha)  # Set alpha for transparency
    plt.xticks([])  
    plt.yticks([])
    
    plt.subplot(1, slots, 1)
    bar = torch.quantile(sal, quantile)
    masked_img = ((sal >= bar).unsqueeze(-1)).numpy() *img
    #img = img * 
    #plt.imshow((stdsal*img).astype(int))  # Set alpha for transparency
    plt.imshow(masked_img)
    plt.xticks([])  
    plt.yticks([])

    
    if with_mask:
        plt.subplot(1, slots, 4)
        msk = mask.unsqueeze(-1).numpy()
        masked_img = ((msk > msk.mean()) *img).astype(int)
        plt.imshow(masked_img)
        plt.xticks([])  
        plt.yticks([])

    if with_pos:
        plt.subplot(1, slots, 4)
        msk = mask.unsqueeze(-1).numpy()
        pos = (osal > 0)
        plt.imshow(pos, cmap='jet')
        plt.xticks([])  
        plt.yticks([])

    if save_path:
        plt.savefig(save_path, dpi=800, bbox_inches='tight', transparent=False, pad_inches=0)
    else:
        plt.show()    

def show_sal_dict(sals, img, mag=False):
    for name, sal in sals.items():
        showsal(sal[0].cpu(), img, caption=name, mag=True)


def show_single_sal(img, allsal, name=None, alpha=None, mag=False, grayscale=False):
    
    pimg=img.resize((224,224))  
    plt.imshow(pimg)  
    plt.xticks([])  
    plt.yticks([])

    if name is None:
        return
    sal = allsal[name]
    #nsal = F.interpolate(sal.unsqueeze(0), scale_factor=int(224 / 7), mode="bilinear")[0]

    nsal = sal
    if 'LTX' in name:
        nsal = torch.sigmoid(nsal)
    nsal = nsal * (nsal >= 0)

        
    nsal = (nsal - nsal.min()) / (nsal.max() - nsal.min())
    if mag:
        #nsal = torch.min(nsal, torch.quantile(sal,0.999))
        nsal = torch.max(nsal, torch.quantile(nsal,0.01))
    nsal = nsal[0]
    
        
    if grayscale:
        plt.imshow(nsal, cmap=plt.cm.gray, vmin=0, vmax=1)
    elif alpha is None:
        plt.imshow(nsal, cmap='jet', alpha=nsal*0.7)  # Set alpha for transparency
    else:
        plt.imshow(nsal, cmap='jet', alpha=alpha)


def show_single_sal_div(img, allsal, name=None, alpha=None, mag=False, grayscale=False):
    
    pimg=img.resize((224,224))  
    plt.imshow(pimg)  
    plt.xticks([])  
    plt.yticks([])

    if name is None:
        return
    sal = allsal[name]
    #nsal = F.interpolate(sal.unsqueeze(0), scale_factor=int(224 / 7), mode="bilinear")[0]

    nsal = sal[0]
    #if 'LTX' not in name:
    #nsal = nsal * (nsal >= 0)

    vmin = nsal.min()
    vmax = nsal.max()
        
    alpha = alpha or 0.7

    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    plt.imshow(nsal, cmap='coolwarm', norm=norm, alpha=alpha)            
    #plt.imshow(nsal, cmap='jet', alpha=alpha)


def show_grid_sals(sals_list, images, method_names, figsize=(10,10), fontsize=7, alpha=None, mag=True, get_method_alias=None):
    if type(images) != list:
        images = [images]  
        sals_list = [sals_list]
    
    idx = 1
    if figsize:
        plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.tight_layout(pad=0)

    nrows = len(images)    
    for row, img in enumerate(images):
        sals = sals_list[row]
        #method_names = list(sals.keys())
        plt.subplot(nrows, len(method_names)+1, idx)         
        idx += 1
        show_single_sal(img, sals, None)
        for cidx, method_name in enumerate(method_names):        
            plt.subplot(nrows, len(method_names)+1, idx) 
            idx += 1
            if get_method_alias:
                alias = get_method_alias(method_name)
            else:
                alias = method_name
            if row == 0:
                if alias == "TCE":
                    alias = "LSC"
                plt.title(alias, fontsize=fontsize)
            show_single_sal(img, sals, method_name, alpha=alpha, mag=mag)
