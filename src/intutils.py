import numpy as np
from matplotlib import pyplot as plt
import torch



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

def showsal(sal, img, caption="", quantile=0.9):
    #stdsal = np.array( ((sal - sal.min()) / (sal.max()-sal.min())).unsqueeze(-1)) 
    #stdsal = (stdsal > 0.7)
    mask = (sal - sal.min()) / (sal.max()-sal.min())
    plt.subplot(1, 4, 1)
    plt.title(caption)
    plt.imshow(sal, cmap='jet')#cmap='RdBu')
    plt.xticks([])  
    plt.yticks([])
    plt.subplot(1, 4, 2)
    plt.imshow(img)    
    plt.imshow(sal, cmap='jet', alpha=0.4)  # Set alpha for transparency
    plt.xticks([])  
    plt.yticks([])
    
    plt.subplot(1, 4, 3)
    bar = torch.quantile(sal, quantile)
    masked_img = ((sal > bar).unsqueeze(-1)).numpy() *img
    #img = img * 
    #plt.imshow((stdsal*img).astype(int))  # Set alpha for transparency
    plt.imshow(masked_img)
    plt.xticks([])  
    plt.yticks([])

    plt.subplot(1, 4, 4)
    
    masked_img = (mask.unsqueeze(-1).numpy() *img).astype(int)
    plt.imshow(masked_img)
    plt.xticks([])  
    plt.yticks([])

    plt.show()    

def show_sal_dict(sals, img):
    for name, sal in sals.items():
        showsal(sal[0].cpu(), img, caption=name)
