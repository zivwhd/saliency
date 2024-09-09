from cpe import *
from tqdm import tqdm
import torch.nn.functional as F

class HexSegments:
    @staticmethod
    def generate_hex_grid(width, height, spacing):
        points = []
        for x in range(0, width, spacing):
            for y in range(0, height, spacing):
                points.append((x, y + (x // spacing % 2) * spacing / 2))
        return np.array(points)
    @staticmethod
    def fill_polygon_in_tensor(tensor, polygon, value=1):
        mask = np.zeros_like(tensor, dtype=np.uint8)
        polygon = np.array(polygon, np.int32)
        cv2.fillPoly(mask, [polygon], 1)
        tensor[mask == 1] = value
        
    @staticmethod
    def create(width, height, spacing):

        padded_width, padded_height = width + spacing *  3, height + spacing * 3
        points = HexSegments.generate_hex_grid(padded_width, padded_height, spacing)
        vor = Voronoi(points)

        data =  np.zeros((padded_height, padded_width), dtype=np.int32)

        for idx, region in enumerate(vor.regions):
            # Check if the region is valid (no vertices at infinity)
            if -1 in region or len(region) == 0:
                continue

            polygon = [vor.vertices[i] for i in region]

            # Fill the selected polygon in the tensor with value 1
            HexSegments.fill_polygon_in_tensor(data, polygon, value=idx + 1)

        return data[spacing : spacing + height, spacing : spacing + width]


def gen_seg_masks_org(segments, nmasks, prob=0.5, width = 224, height = 224):

    masks = np.zeros((nmasks, height, width), dtype=np.float32)
    for idx in range(nmasks):#(32 masks)                            
        w_crop = np.random.randint(0, segments.shape[0] - width)
        h_crop = np.random.randint(0, segments.shape[1] - height)
    
        wseg = segments[h_crop:height + h_crop, w_crop:width + w_crop]
        items = np.unique(wseg)
        nitems = items.shape[0]
        
        selection = (np.random.random(nitems) > prob)
        
        for sid in items[selection]:
            masks[idx][wseg == sid] = 1
    return masks

class ProgGen(IpwGen):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.step_size = 160
        self.start = 160
        self.num_masks = 0
        self.drop = None
        self.drop_prob = 1.0
        self.drop_quantile = 0.995
        self.step_quantile = 0.005
        self.vr = True
        
    def gen_masks(self, batch_size):
        exp_masks =  self.mgen.gen_masks(batch_size)

        if self.drop is not None:
            drop = (torch.rand(batch_size) < self.drop_prob).unsqueeze(1).unsqueeze(1) * self.drop.unsqueeze(0)
            exp_masks = exp_masks * (1-drop).numpy()

        if self.num_masks > self.start and  (self.num_masks % self.step_size) + batch_size >= self.step_size :            
            sal = self.get_ate_sal()
            bar = torch.quantile(sal, self.drop_quantile)
            self.drop_quantile -= self.step_quantile
            drop = (sal > bar)
            mshape = exp_masks.shape[-2:]
            self.drop = torch.zeros(mshape)
            if self.vr:
                print(exp_masks.shape, self.drop.shape, self.pad, mshape, drop.sum())
                self.vr = False
            
            self.drop[self.pad:self.pad+self.ishape[0], self.pad:self.pad+self.ishape[1]] = 1.0 * drop

        self.num_masks += batch_size
        return exp_masks


class GradGen(IpwGen):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

        self.step_size = 64
        self.start = 64 
        self.num_masks = 0

        self.normsal = None

        self.drop = None
        self.drop_prob = 1.0
        self.drop_quantile = 0.995
        self.step_quantile = 0.005
        self.vr = True
        
    def gen_masks(self, batch_size):
        exp_masks =  self.mgen.gen_masks_cont(batch_size)


        if self.normsal is not None:            
            exp_masks = exp_masks < self.normsal
        else:
            exp_masks = (exp_masks < 0.5)

        if self.num_masks > self.start and  (self.num_masks % self.step_size) + batch_size >= self.step_size :            

            sal = self.get_ate_sal()[0]
            normsal = 0.1+0.5*((sal - sal.min()) / (sal.max() - sal.min()))
            mshape = exp_masks.shape[-2:]
            
            self.normsal = np.zeros((1,*mshape), dtype=np.float32)
            #print("### ###", mshape,sal.shape, exp_masks.shape, self.normsal.shape)
            self.normsal[:,:] = 0.5
            self.normsal[0, self.pad:self.pad+self.ishape[0],self.pad:self.pad+self.ishape[1]] = normsal
            #### PUSH_ASSERT

        self.num_masks += batch_size
        #raise Exception(f"AAA {exp_masks.dtype}")
        return exp_masks

class IpwComb(IpwGenBase):
    def __init__(self, parts, active=True):
        self.saliency = None
        self.weights = None
        self.parts = parts
        self.active = active

    def gen(self, model, inp, nmasks, batch_size=32, **kwargs):
        if self.active:
            for part in self.parts:
                part.gen(model, inp, nmasks, batch_size=batch_size, **kwargs)
        self.update()

    def update(self):
        self.saliency = None
        self.weights = None
        for part in self.parts:
            if self.saliency is None:
                self.saliency = part.saliency.clone()
                self.weights = part.weights.clone()
            else:
                self.saliency += part.saliency
                self.weights += part.weights


class HexMaskGen:

    def __init__(self, segsize, mshape, efactor=4):
        self.segments = HexSegments.create(mshape[0] * efactor, mshape[1] * efactor, segsize)
        self.mshape = mshape
    
    def gen_masks(self, nmasks):
        return gen_seg_masks(self.segments, nmasks, width=self.mshape[1], height=self.mshape[0])

class IpwGenHex(IpwGen):

    def __init__(self, segsize=68, ishape = (224,224), MaskGen=HexMaskGen, degrees=[0,60,120,180,240,300]):
        super().__init__(segsize=segsize, ishape=ishape, MaskGen=MaskGen, degrees=degrees)


###
class SimpGen:
    def __init__(self, segsize=68, ishape = (224,224), force_mask=None, collect_masks=False):
        self.treatment = None
        self.ctrl = None
        self.treatment2 = None
        self.ctrl2 = None
        self.total_masks = 0        
        self.weights = None

        self.segsize = segsize
        self.pad = self.segsize // 2
        self.ishape = ishape
        self.mgen = SqMaskGen(segsize, mshape=ishape)
        self.weights = torch.zeros(ishape)
        self.sals = torch.zeros(ishape)
        self.sals2 = torch.zeros(ishape)
        self.force_mask = force_mask

        self.collect_masks = collect_masks
        self.all_masks = []
        self.all_pred = []

    def gen_(self, model, inp, itr=125, batch_size=32):
        
        h = self.ishape[0]
        w = self.ishape[1]
        pad = self.pad

        force_mask = self.force_mask
        if force_mask is not None:
            force_mask = force_mask.unsqueeze(0).to(inp.device)

        for idx in tqdm(range(itr)):
            masks = self.mgen.gen_masks(batch_size)
            self.total_masks += masks.shape[0]
            #if type(exp_masks) == np.ndarray:
            #    exp_masks = torch.tensor(exp_masks)
            #masks = masks.unsqueeze(1)            
            dmasks = masks.to(inp.device)    
            if force_mask is not None:
                dmasks = dmasks | force_mask

            out = masked_output(model, inp, dmasks)
            mout = out.unsqueeze(-1).unsqueeze(-1)

            if self.collect_masks:
                self.all_masks.append(masks.cpu())
                self.all_pred.append(mout.cpu())
            streatment = mout * dmasks.unsqueeze(1)
            sctrl = mout * (1 - 1.0 * dmasks.unsqueeze(1))

            treatment = streatment.sum(dim=0)
            ctrl = sctrl.sum(dim=0)

            treatment2 = (streatment*streatment).sum(dim=0)
            ctrl2 = (sctrl*sctrl).sum(dim=0)
              
            weights = dmasks.sum(dim=0, keepdim=True)

            if self.treatment is None:
                self.treatment = treatment
                self.ctrl = ctrl
                self.treatment2 = treatment2
                self.ctrl2 = ctrl2
                self.weights = weights
            else:
                self.treatment += treatment
                self.ctrl += ctrl
                self.treatment2 += treatment2
                self.ctrl2 += ctrl2
                self.weights += weights
            

    def gen(self, model, inp, nmasks, batch_size=32, **kwargs):        
        with torch.no_grad():
            self.gen_(model=model, inp=inp, itr=nmasks//batch_size, batch_size=batch_size, **kwargs)
            if nmasks % batch_size:
                self.gen_(model=model, inp=inp, itr=1, batch_size=nmasks % batch_size, **kwargs)

    def var(self, values, values2, weights):
        return (values2 / weights) - (values * values) / (weights * weights)
    
    def get_ate_sal_i(self):
        ctrl_weights = (self.total_masks - self.weights)
        ate = (self.treatment /  self.weights) - (self.ctrl / ctrl_weights)
        
        treatment_var = self.var(self.treatment, self.treatment2, self.weights)
        ctrl_var = self.var(self.ctrl, self.ctrl2, ctrl_weights)
        ate_var = treatment_var + ctrl_var

        return ate, ate_var

    def get_ate_sal(self):
        return self.get_ate_sal_i()[0]

def gsobel(K):    
    assert K % 2 == 1
    arng = torch.arange(K, dtype=torch.float32)
    offs = arng-arng.mean()
    dist = offs.abs()    
    return torch.nan_to_num(offs / (dist.unsqueeze(0)**2 +  dist.unsqueeze(1)**2),0)

sblx = gsobel(31)
sbly = sblx.transpose(0,1)

class RelIpwGen:
    def __init__(self, segsize=64, ishape = (224,224)):
        super(self).__init(segsize=segsize, ishape=ishape, force_mask=None, collect_masks=True)
        
        sobelK = ((segsize // 2) - 1 + (segsize // 2) % 2)
        self.rdist = sobelK
        self.sblx = gsobel(sobelK)
        self.sbly = sblx.transpose(0,1)

    def get_ips_sal(self, clip=0.1):
        
        ## first get simple ate sal - to find potential confounder
        ate = self.get_ate_sal()
        isal = ate.cpu()

        ## find saliency gradient - the potnetial confounder is upward
        ctx = F.conv2d(isal.unsqueeze(0), sblx.unsqueeze(0).unsqueeze(0), padding="same")[0,0]
        cty = F.conv2d(isal.unsqueeze(0), sbly.unsqueeze(0).unsqueeze(0), padding="same")[0,0]
        csz = (ctx**2 + cty**2).sqrt()
        cszbar = torch.quantile(csz, 0.01)

        ## calculate offsets 
        offsx = (ctx*dist/csz).to(torch.int32)
        offsy = (cty*dist/csz).to(torch.int32)

        # offset to indexes, validation
        H, W = self.ishape
        idxx = offsx + torch.arange(W).unsqueeze(0)
        idxy = offsy + torch.arange(H).unsqueeze(0)
        isok = ((csz > cszbar) & (idxx >= 0) & (idxx < W) & (idxy >= 0) & (idxy < H))
        
        ## flatten indexes
        confidx = torch.maximum(torch.minimum(idxx.flatten()+idxy.flatten()*H, torch.tensor(H*W-1)), torch.tensor([0]))

        icats = torch.arange(4).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gidx = None
        prev_bmask_shape = None
        s_saliency = None
        s_weights = None

        # iterate over masks, and acummulate alienct, weights in bins
        for idx, bmasks in enumerate(simp.all_masks):
    
            if bmasks.shape != prev_bmask_shape:
                gidx = ((torch.arange(bmasks.shape[0]) * confidx.numel()).unsqueeze(1) + confidx.unsqueeze(0)).flatten()
                prev_bmask_shape = bmasks.shape
                
            ref = bmasks.flatten().gather(0, gidx).reshape(bmasks.shape)
            mout = simp.all_pred[idx]
            #print(mout)
            #print(bmasks.shape)
            ref = bmasks.flatten().gather(0, gidx).reshape(bmasks.shape)
            cat = ((bmasks > 0.5) * 1 + (ref > 0.5) * 2).unsqueeze(0)
            
            mbin = (icats == cat)
            #print(cat.shape, mout.shape, mbin.shape)
            saliency = (mout.squeeze(1).unsqueeze(0) * mbin).sum(dim=1)
            print(mbin.shape)
            weights = mbin.sum(dim=1)    
            if s_saliency is None:
                s_saliency = saliency
                s_weights = weights
            else:
                s_saliency += saliency
                s_weights += weights

        cidx = torch.arange(4)
        treatment_sal = s_saliency[(cidx & 1) == 1]  ## dim:0 is of size 2
        control_sal = s_saliency[(cidx & 1) == 0]

        p_treatment_weights = s_weights[(cidx & 1) == 1]
        p_control_weights = s_weights[(cidx & 1) == 0]
        
        t_clipping = torch.tensor([clip])    
        c_clipping = torch.tensor([clip])    

        treatment_prob =  torch.max(p_treatment_weights / (p_treatment_weights + p_control_weights + 1), t_clipping)
        control_prob =  torch.max(p_control_weights / (p_treatment_weights + p_control_weights + 1), c_clipping)

        ipw = (
            ((treatment_sal / treatment_prob).sum(dim=0) / (p_treatment_weights / treatment_prob).sum(dim=0)) -
            ((control_sal / control_prob).sum(dim=0) / (p_control_weights / control_prob).sum(dim=0)) ).unsqueeze(0)            

        return ipw
#treatment_prob.shape, p_treatment_weights.shape, p_control_weights.shape, t_clipping.shape


