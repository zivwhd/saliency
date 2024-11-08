import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys, logging
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def numpy_to_torch(img, requires_grad = True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad = requires_grad)
    return v

def median_blur(input_tensor, kernel_size):
    # Pad the tensor to handle edges
    padding = kernel_size // 2
    input_padded = F.pad(input_tensor, (padding, padding, padding, padding), mode='reflect')

    # Unfold the tensor to get sliding windows of the given kernel size
    unfolded = input_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)

    # Compute the median across the kernel dimensions
    median_filtered = unfolded.contiguous().view(*unfolded.shape[:4], -1).median(dim=-1)[0]    
    return median_filtered


class IEMPertSaliencyCreator:
    def __init__(self, out_coeff=1.0, tv_beta=1, tv_coeff=0.2, l1_coeff=0.005, iterations=300,
                 learning_rate=0.1):        
        self.out_coeff = out_coeff
        self.tv_beta = tv_beta
        self.tv_coeff = tv_coeff
        self.l1_coeff = l1_coeff
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.blur = False

    def __call__(self, me, inp, catidx):   
        desc = f"IEMPert_{self.iterations}_o{self.out_coeff}_tv{self.tv_coeff}_{self.tv_beta}_l{self.l1_coeff}"        
        sal = self.explain(me, inp, catidx)        
        return {desc : sal}
    
    def explain(self, me, inp, catidx):
    
        #Hyper parameters. 
        #TBD: Use argparse
        tv_beta = self.tv_beta
        learning_rate = self.learning_rate
        max_iterations = self.iterations #300 #500
        l1_coeff = self.l1_coeff ## 0.01
        tv_coeff = self.tv_coeff # 0.02 ##0.2        
        out_coeff = self.out_coeff
        device = inp.device
        
        print(f"itr={self.iterations}; tv={self.tv_coeff}:{self.tv_beta}; l1={self.l1_coeff}; lr={learning_rate}")
        #model = load_model()
        #original_img = cv2.imread(sys.argv[1], 1)
        #original_img = cv2.resize(original_img, (224, 224))
        #img = np.float32(original_img) / 255
        #fmdl = me.narrow_model(catidx, with_softmax=False)
        smdl = me.narrow_model(catidx, with_softmax=True)
        
        gaussian_blur = T.GaussianBlur(kernel_size=(11, 11), sigma=5.0)
        
        # Apply the Gaussian blur to the tensor
        
        if self.blur:
            baseline = gaussian_blur(inp)
        else:
            baseline = torch.zeros(inp.shape).to(inp.device)
        #blurred_img2 = median_blur(inp, 11)
        
        #iimg = inp.cpu().squeeze(0).numpy().transpose(1,2,0)
        #print(iimg.shape)
        #blurred_img1 = cv2.GaussianBlur(iimg, (11, 11), 5)
        #blurred_img2 = np.float32(cv2.medianBlur(iimg, 11))
        #blurred_img = (blurred_img1 + blurred_img2) / 2
        #blurred_img_numpy = blurred_img.cpu().numpy()

        #mask_init = np.ones((28, 28), dtype = np.float32) + torch.randn(28,28) * 0.1
        
        
        # Convert to torch variables
        #img = preprocess_image(img)
        #blurred_img = blurred_img2 #preprocess_image(blurred_img2)
        
        mask = torch.ones((1, 1, 28, 28), dtype = torch.float32, requires_grad=True, device=inp.device)
        #mask = torch.randn(1,1,28,28).to(inp.device) * 0.1 + 1.0
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).to(inp.device)
        optimizer = torch.optim.Adam([mask], lr=learning_rate)

        #target = torch.nn.Softmax()(model(img))
        #category = np.argmax(target.cpu().data.numpy())
        #print "Category with highest probability", category
        #print "Optimizing.. "

        for i in range(max_iterations):
            upsampled_mask = upsample(mask)

            perturbated_input = (
                inp * upsampled_mask + baseline * (1-upsampled_mask))

            noise = torch.normal( mean=0, std=0.2, size=inp.shape).to(inp.device)
            
            perturbated_input += noise
            #outputs = torch.nn.Softmax()(model(perturbated_input))
            outputs = smdl(perturbated_input)

            loss = (
                l1_coeff*torch.mean(torch.abs(1 - mask)) + 
                tv_coeff*tv_norm(mask, tv_beta) + 
                out_coeff * outputs
            )

            idesc = f"Itr {i}: loss={loss}"
            if i % 10 == 0:
                logging.info(idesc)
                print(idesc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional: clamping seems to give better results
            mask.data.clamp_(0, 1)

        upsampled_mask = upsample(mask)
        sal = upsampled_mask.detach().cpu().squeeze(0)
        sal = (sal - sal.min()) / (sal.max()- sal.min())        
        return 1 - sal
