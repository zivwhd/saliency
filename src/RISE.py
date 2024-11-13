import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from reports import report_duration
import time

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy', with_tqdm=False):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        if with_tqdm:
            sq = tqdm
        else:
            sq = lambda x, **kwargs: x

        for i in sq(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        if savepath:
            np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x, normalize=True):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        if normalize:
            sal = sal / N / self.p1
        return sal
    
    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal

def RISE_explain(model, inp, itr, seg=8, p1 = 0.5, batch_size=32):
    input_size = inp.shape[-2:]
    rise = RISE(model, input_size, gpu_batch=100)
    sal = None
    N = 0
    for itr in tqdm(range(itr)):
        rise.generate_masks(batch_size, seg, p1, savepath=None, with_tqdm=False)
        N += batch_size
        dsal = rise(inp, normalize = False)
        if sal is None:
            sal = dsal.detach().clone()
        else:
            sal += dsal
    sal = sal / N / p1
    return sal

class RiseSaliencyCreator:

    def __init__(self, nmasks=4000, seg=7, p1=0.5):

        self.nmasks = nmasks
        self.seg = seg
        self.p1 = p1

    def __call__(self, me, inp, catidx, batch_size=32):
        assert self.nmasks % batch_size == 0
        start_time = time.time()
                
        sal = RISE_explain(
            me.narrow_model(catidx), inp, 
            itr = self.nmasks // batch_size, 
            seg=self.seg, p1=self.p1, batch_size=batch_size)
        report_duration(start_time, me.arch, "RISE", self.nmasks)
        name = f"RISE_{self.nmasks}_{self.seg}_{self.p1}"
        return {name : sal.cpu()}
        
