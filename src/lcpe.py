import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import logging, time, pickle
from cpe import SqMaskGen

tqdm = lambda x: x

class LoadMaskGen:
    def __init__(self, path):
        with open(path, "rb") as mf:
            self.masks = pickle.load(mf).all_masks
            self.idx = 0
        
    def gen_masks(self, batch_size):
        if self.idx + batch_size > self.masks.shape[0]:
            raise Exception("ran out of masks")
        rv = self.masks[self.idx:self.idx+batch_size]
        self.idx += batch_size
        return rv



class MaskedRespGen:
    def __init__(self, segsize=48, ishape = (224,224),
                 mgen=None, baseline=None):

        self.segsize = segsize
        self.ishape = ishape
        if mgen is None:
            self.mgen = SqMaskGen(segsize, mshape=ishape)
        else:
            self.mgen = mgen

        if baseline is None:
            self.baseline = torch.zeros(ishape)
        else:
            self.baseline = baseline

        self.all_masks = []
        self.all_pred = []

    def gen_(self, model, inp, itr=125, batch_size=32):
        
        h = self.ishape[0]
        w = self.ishape[1]
        
        baseline = self.baseline.to(inp.device)

        for idx in tqdm(range(itr)):            
            masks = self.mgen.gen_masks(batch_size)
            dmasks = masks.to(inp.device).float()

            pert_inp = inp * dmasks.unsqueeze(1) + baseline * (1.0-dmasks.unsqueeze(1))
            out = model(pert_inp) ## CHNG
            mout = out.unsqueeze(-1).unsqueeze(-1)
            
            self.all_masks.append(masks.cpu())
            self.all_pred.append(mout.cpu())


    def gen(self, model, inp, nmasks, batch_size=32, **kwargs):        
        with torch.no_grad():
            self.gen_(model=model, inp=inp, itr=nmasks//batch_size, batch_size=batch_size, **kwargs)
            if nmasks % batch_size:
                self.gen_(model=model, inp=inp, itr=1, batch_size=nmasks % batch_size, **kwargs)



class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        # Calculate the variation in the x and y directions
        x_diff = torch.abs(x[ :, :-1] - x[ :, 1:])
        y_diff = torch.abs(x[ :-1, :] - x[ 1:, :])
        # Sum the variations to get the total variation loss
        loss = torch.mean(x_diff) + torch.mean(y_diff)
        return loss

class MaskedExplanationSum(nn.Module):
    def __init__(self, H=224, W=224, initial_value=None,):
        super(MaskedExplanationSum, self).__init__()
        # Initialize explanation with given initial value or zeros
        if initial_value is not None:
            self.explanation = nn.Parameter(initial_value)
        else:
            self.explanation = nn.Parameter(torch.zeros(H, W))

    def forward(self, x):
        y =  (x * self.explanation).flatten(start_dim=1).sum(dim=1)
        return y

    def normalize(self, score):
        # Normalize explanation so that its sum equals the provided score
        with torch.no_grad():  # No gradient update needed for normalization
            self.explanation *= score / self.explanation.sum()

# Define the training function
def optimize_explanation_i(
        fmdl, inp, mexp, data, targets, epochs=10, lr=0.001, score=1.0, 
        c_mask_completeness=1.0, c_smoothness=0.1, c_completeness=0.0, c_selfness=0.0,
        c_concentration=0,
        c_tv=0, avg_kernel_size=(5,5),
        c_model=0,
        renorm=False, baseline=None):
    mse = nn.MSELoss()  # Mean Squared Error loss
    tv = TotalVariationLoss()

    if baseline is None:
        assert False ## no default
        baseline = torch.zeros(inp.shape).to(inp.device)

    #print(list(model.parameters()))
    logging.debug(f"### lr={lr}; c_completeness={c_completeness}; c_tv={c_tv}; c_smoothness={c_smoothness}; avg_kernel_size={avg_kernel_size}")
    print(f"### lr={lr}; c_completeness={c_completeness}; c_tv={c_tv}; c_smoothness={c_smoothness}; c_concentration={c_concentration}; avg_kernel_size={avg_kernel_size}")
    optimizer = optim.Adam(mexp.parameters(), lr=lr)
    mexp.normalize(score)
    mexp.train()
    avg_kernel = torch.ones((1,) + avg_kernel_size).to(data.device)
    avg_kernel = avg_kernel / avg_kernel.numel()
    for epoch in range(epochs):
                
        # Forward pass
        optimizer.zero_grad()
        output = mexp(data)
        explanation = mexp.explanation
        comp_loss = mse(output/explanation.numel(), targets/explanation.numel())

        if c_completeness != 0:            
            explanation_sum = mexp.explanation.sum()
            explanation_loss = mse(explanation_sum/explanation.numel(), score/ explanation.numel())             
        else:
            explanation_loss = 0
        
        conv_loss = 0
        if c_smoothness != 0:                
            sexp = F.conv2d(explanation.unsqueeze(0), avg_kernel.unsqueeze(0),padding="same").squeeze()
            conv_loss = mse(explanation, sexp)            
        else:
            conv_loss = 0

        if c_tv != 0:
            tv_loss = tv(explanation)
        else:
            tv_loss = 0

        if c_model:
            explanation_mask = (explanation - explanation.min()) / (explanation.max() - explanation.min())
            masked_inp = explanation_mask * inp + (1-explanation_mask) * baseline
            prob = fmdl(masked_inp)
            model_loss = -torch.log(prob)
        else:
            model_loss = 0

        if c_concentration != 0:
            #norm_explanation = (explanation / torch.sqrt(explanation*explanation))
            norm_explanation = (explanation / explanation.sum())
            concentration_loss = - ((norm_explanation ** 2).sum())
        else:
            concentration_loss = 0

        ## tar loss
        if c_selfness != 0:
            mask_size = data.flatten(start_dim=1).sum(dim=1)
            exp_val = data * targets.unsqueeze(1).unsqueeze(1) / mask_size.unsqueeze(1).unsqueeze(1)
            act_val = data * explanation.unsqueeze(0)
            tar_loss = mse(exp_val, act_val)
        else:
            tar_loss = 0

        # Backward pass and optimization        
        total_loss = (
            c_mask_completeness * comp_loss + 
            c_smoothness * conv_loss + 
            c_completeness * explanation_loss + 
            c_tv * tv_loss +
            c_selfness * tar_loss +
            c_model * model_loss + 
            c_concentration * concentration_loss
            )
        
        total_loss.backward()        
        optimizer.step()

        # Normalize the explanation with the given score
        if epoch % 100 == 0 and renorm:
            mexp.normalize(score)
        
        pdesc = (f"Epoch {epoch+1}/{epochs} Loss={total_loss.item()}; ES={explanation.sum()}; comp_loss={comp_loss}; "
                f"exp_loss={explanation_loss}; conv_loss={conv_loss}; tv_loss={tv_loss}; model_loss={model_loss};"
                f"concentration_loss={concentration_loss}")
        logging.debug(pdesc)
        print(pdesc)


def optimize_explanation(fmdl, inp, initial_explanation, data, targets, score=1.0, 
                         epochs=0, model_epochs=0, c_model=0,
                         **kwargs):
    # Initialize the model with the given initial explanation
    mexp = MaskedExplanationSum(initial_value=initial_explanation)
    mexp = mexp.to(data.device)
    mexp.normalize(score)    
    # Train the model by passing all additional arguments through kwargs
    start_time = time.time()
    optimize_explanation_i(fmdl, inp, mexp, data, targets, score=score, c_model=0, epochs=epochs-model_epochs, **kwargs)
    mid_time = time.time()
    logging.info(f"Optimization I: {mid_time - start_time}")    
    print(f"Optimization I: {mid_time -start_time}")    
    if model_epochs:
        optimize_explanation_i(fmdl, inp, mexp, data, targets, score=score, c_model=c_model, epochs=model_epochs, **kwargs)
    end_time = time.time()
    logging.info(f"Optimization Done: ({mid_time - start_time}) , ({end_time - start_time})")    
    print(f"Optimization Done: ({mid_time - start_time}) , ({end_time - start_time})")    
    mexp.normalize(score)
    # Return the updated explanation parameter
    return mexp.explanation.detach()


class MaskedRespData:
    def __init__(self, baseline_score, label_score, added_score, all_masks, all_pred, baseline):
        self.baseline_score = baseline_score
        self.label_score = label_score
        self.added_score = added_score
        self.all_masks = all_masks
        self.all_pred = all_pred
        self.baseline = baseline


class ZeroBaseline:

    def __init__(self):
        pass

    def __call__(self, inp):
        return torch.zeros(inp.shape).to(inp.device)
    
    @property
    def desc(self):
        return "Zr"
    
class RandBaseline:

    def __init__(self):
        pass

    def __call__(self, inp):
        return torch.normal(0.5, 0.25, size=inp.shape).to(inp.device)    
    
    @property
    def desc(self):
        return "Rnd"

class BlurBaseline:

    def __init__(self, ksize=31, sigma=17.0):
        self.ksize = ksize
        self.sigma = sigma
        self.gaussian_blur = T.GaussianBlur(kernel_size=(ksize, ksize), sigma=sigma)        

    def __call__(self, inp):
        return self.gaussian_blur(inp)
            
    @property
    def desc(self):
        return "Blr{self.ksize}x{self.sigma}"

class CompExpCreator:

    def __init__(self, nmasks=500, segsize=64, batch_size=32, 
                 lr = 0.05, c_mask_completeness=1.0, c_completeness=0.1, 
                 c_smoothness=0, c_selfness=0.0, c_tv=1,
                 c_concentration=0,
                 avg_kernel_size=(5,5),
                 epochs=200, 
                 model_epochs=200, c_model=0,
                 mgen=None,
                 desc = "MComp",
                 baseline_gen = ZeroBaseline(),                 
                 **kwargs):
        
        self.segsize = segsize
        self.nmasks = nmasks
        self.batch_size = batch_size
        self.c_completeness = c_completeness
        self.c_smoothness = c_smoothness  
        self.c_tv = c_tv
        self.c_selfness = c_selfness
        self.c_mask_completeness = c_mask_completeness
        self.c_model = c_model
        self.c_concentration = c_concentration
        self.lr = lr
        self.avg_kernel_size = avg_kernel_size      
        self.epochs = epochs
        self.model_epochs = model_epochs
        self.desc = desc
        self.mgen = mgen
        self.baseline_gen = baseline_gen
        if self.model_epochs == 0 or self.c_model == 0:
            self.model_epochs = 0
            self.c_model = 0


    def description(self):
        desc = f"{self.desc}_{self.nmasks}_{self.segsize}_{self.epochs}"
                                
        if self.model_epochs:
            desc += f":{self.model_epochs}"

        if self.c_smoothness != 0:
            desc += f"_krn{self.c_smoothness}_" + str("x").join(map(str, self.avg_kernel_size))

        if self.c_mask_completeness:
            desc += f"_msk{self.c_mask_completeness}"

        if self.c_completeness:
            desc += f"_cp{self.c_completeness}"

        if self.c_tv:
            desc += f"_tv{self.c_tv}"

        if self.c_model:
            desc += f"_mdl{self.c_model}"

        if self.c_selfness:
            desc += f"_sf{self.c_selfness}"

        return desc
    
    def __call__(self, me, inp, catidx, data=None):
        desc = self.description()
        sal = self.explain(me, inp, catidx, data=data)
        csal = sal.cpu().unsqueeze(0)

        return {desc : csal}
        #return {
        #    f"{self.desc}_{self.nmasks}_{self.segsize}{ksdesc}_{self.epochs}_{self.c_completeness}_{self.c_smoothness}" : sal.cpu().unsqueeze(0)
        #}

    def generate_data(self, me, inp, catidx):
        baseline = self.baseline_gen(inp)
        mgen = MaskedRespGen(self.segsize, mgen=self.mgen, baseline=baseline)
        fmdl = me.narrow_model(catidx, with_softmax=True)
        logging.debug(f"generating {self.nmasks} masks and responses")
        mgen.gen(fmdl, inp, self.nmasks, batch_size=self.batch_size,)        
        logging.debug("Done generating masks")

        rfactor = inp.numel()        
        baseline_score = fmdl(baseline).detach().squeeze() * rfactor
        label_score = fmdl(inp).detach().squeeze() * rfactor

        added_score = label_score - baseline_score
        #print(label_score, baseline_score, delta_score)
    
        device = me.device
        all_masks = torch.concat(mgen.all_masks).to(device)
        all_pred = torch.concat(mgen.all_pred).to(device).squeeze() * rfactor - baseline_score
        all_pred.shape, baseline_score.shape
        return MaskedRespData(
            baseline_score = baseline_score,
            label_score = label_score,
            added_score = added_score,
            all_masks = all_masks,
            all_pred = all_pred,
            baseline = baseline
        )

    def explain(self, me, inp, catidx, data=None, initial=None):

        if data is None:
            data = self.generate_data(me, inp, catidx)
        if initial is None:
            #initial = torch.rand(inp.shape[-2:]).to(inp.device)
            initial = (torch.randn(224,224)*0.2+1).abs()

        fmdl = me.narrow_model(catidx, with_softmax=True)        

        sal = optimize_explanation(fmdl, inp, initial, data.all_masks, data.all_pred, score=data.added_score, 
                                   epochs=self.epochs, model_epochs=self.model_epochs, lr=self.lr, avg_kernel_size=self.avg_kernel_size,
                                   c_completeness=self.c_completeness, c_smoothness=self.c_smoothness, 
                                   c_tv=self.c_tv, c_selfness=self.c_selfness,
                                   c_mask_completeness=self.c_mask_completeness,
                                   c_model=self.c_model,
                                   c_concentration=self.c_concentration,
                                   baseline=data.baseline)
        
        return sal



class MultiCompExpCreator:

    def __init__(self, nmasks=500, segsize=64, batch_size=32, baselines=[ZeroBaseline()],
                 groups=[]):
        self.nmasks = nmasks
        self.segsize=segsize
        self.batch_size = batch_size
        self.baselines = baselines
        self.groups = groups

    def __call__(self, me, inp, catidx):
        all_sals = {}
        for bgen in self.baselines:
            desc = "MComp" + bgen.desc
            dc = CompExpCreator(nmasks=self.nmasks, segsize=self.segsize, batch_size=self.batch_size,
                                baseline_gen=bgen
                                )
            data = dc.generate_data(me, inp, catidx)
            
            for kwargs in self.groups:
                algo = CompExpCreator(nmasks=self.nmasks, segsize=self.segsize, batch_size=self.batch_size, 
                                      desc=desc, **kwargs)
                res = algo(me, inp, catidx, data=data)
                all_sals.update(res)
        logging.info(f"generated: {list(all_sals.keys())}")
        return all_sals

