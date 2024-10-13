import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from cpe import SqMaskGen

tqdm = lambda x: x

class MaskedRespGen:
    def __init__(self, segsize=48, ishape = (224,224)):

        self.segsize = segsize
        self.ishape = ishape
        self.mgen = SqMaskGen(segsize, mshape=ishape)

        self.all_masks = []
        self.all_pred = []

    def gen_(self, model, inp, itr=125, batch_size=32):
        
        h = self.ishape[0]
        w = self.ishape[1]


        for idx in tqdm(range(itr)):
            masks = self.mgen.gen_masks(batch_size)
            dmasks = masks.to(inp.device)    

            out = model(inp * dmasks.unsqueeze(1)) ## CHNG
            mout = out.unsqueeze(-1).unsqueeze(-1)

            
            self.all_masks.append(masks.cpu())
            self.all_pred.append(mout.cpu())

    def gen(self, model, inp, nmasks, batch_size=32, **kwargs):        
        with torch.no_grad():
            self.gen_(model=model, inp=inp, itr=nmasks//batch_size, batch_size=batch_size, **kwargs)
            if nmasks % batch_size:
                self.gen_(model=model, inp=inp, itr=1, batch_size=nmasks % batch_size, **kwargs)



class MaskPredict(nn.Module):
    def __init__(self, H=224, W=224, initial_value=None,):
        super(MaskPredict, self).__init__()
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
def optimize_explanation_i(model, data, targets, epochs=10, lr=0.001, score=1.0, 
                c_mask_completeness=1.0, c_smoothness=0.1, c_completeness=0.0, c_selfness=0.0,
                avg_kernel_size=(5,5),
                renorm=False,
                ):
    criterion = nn.MSELoss()  # Mean Squared Error loss
    #print(list(model.parameters()))
    logging.debug(f"### lr={lr}; c_completeness={c_completeness}; c_smoothness={c_smoothness}; avg_kernel_size={avg_kernel_size}")
    print(f"### lr={lr}; c_completeness={c_completeness}; c_smoothness={c_smoothness}; avg_kernel_size={avg_kernel_size}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.normalize(score)
    model.train()
    avg_kernel = torch.ones((1,) + avg_kernel_size).to(data.device)
    avg_kernel = avg_kernel / avg_kernel.numel()
    for epoch in range(epochs):
                
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        exp = model.explanation
        comp_loss = criterion(output/exp.numel(), targets/exp.numel())

        if c_completeness != 0:            
            explanation_sum = model.explanation.sum()
            explanation_loss = criterion(explanation_sum/exp.numel(), score/ exp.numel())             
        else:
            explanation_loss = 0

        
        conv_loss = 0
        if c_smoothness != 0:                
            sexp = F.conv2d(exp.unsqueeze(0), avg_kernel.unsqueeze(0),padding="same").squeeze()
            conv_loss = criterion(exp, sexp)            
        else:
            conv_loss = 0

        ## tar loss
        if c_selfness != 0:
            mask_size = data.flatten(start_dim=1).sum(dim=1)
            exp_val = data * targets.unsqueeze(1).unsqueeze(1) / mask_size.unsqueeze(1).unsqueeze(1)
            act_val = data * exp.unsqueeze(0)
            tar_loss = criterion(exp_val, act_val)
        else:
            tar_loss = 0

        # Backward pass and optimization        
        total_loss = (c_mask_completeness * comp_loss + c_smoothness * conv_loss + 
                      c_completeness * explanation_loss + c_selfness * tar_loss)
        
        total_loss.backward()        
        optimizer.step()

        # Normalize the explanation with the given score
        if epoch % 100 == 0 and renorm:
            model.normalize(score)
        
        logging.debug(f"Epoch {epoch+1}/{epochs} Loss={total_loss.item()}; ES={model.explanation.sum()}; comp_loss={comp_loss}; exp_loss={explanation_loss}; conv_loss={conv_loss}")
        print(f"Epoch {epoch+1}/{epochs} Loss={total_loss.item()}; ES={model.explanation.sum()}; tar_loss={tar_loss}; comp_loss={comp_loss}; exp_loss={explanation_loss}; conv_loss={conv_loss}")


def optimize_explanation(initial_explanation, data, targets, score=1.0, **kwargs):
    # Initialize the model with the given initial explanation
    model = MaskPredict(initial_value=initial_explanation)
    model = model.to(data.device)
    model.normalize(score)
    iexp = model.explanation.data.clone().detach()
    # Train the model by passing all additional arguments through kwargs
    optimize_explanation_i(model, data, targets, score=score, **kwargs)
    
    # Return the updated explanation parameter
    return model.explanation.detach()


class MaskedRespData:
    def __init__(self, baseline_score, label_score, added_score, all_masks, all_pred):
        self.baseline_score = baseline_score
        self.label_score = label_score
        self.added_score = added_score
        self.all_masks = all_masks
        self.all_pred = all_pred

class CompExpCreator:

    def __init__(self, nmasks=500, segsize=48, batch_size=32, 
                 lr = 0.05, c_mask_completeness=1.0, c_completeness=0, 
                 c_smoothness=1.0, c_selfness=0.0, 
                 avg_kernel_size=(5,5),
                 epochs=500, desc = "CompRd",
                 **kwargs):
        self.segsize = segsize
        self.nmasks = nmasks
        self.batch_size = batch_size
        self.c_completeness = c_completeness
        self.c_smoothness = c_smoothness  
        self.c_selfness = c_selfness
        self.c_mask_completeness = c_mask_completeness
        self.lr = lr
        self.avg_kernel_size = avg_kernel_size      
        self.epochs = epochs
        self.desc = desc

    def __call__(self, me, inp, catidx):
        sal = self.explain(me, inp, catidx)
        ksdesc = str("x").join(map(str, self.avg_kernel_size))
        return {
            f"{self.desc}_{self.nmasks}_{self.segsize}_{ksdesc}_{self.epochs}_{self.c_completeness}_{self.c_smoothness}" : sal.cpu().unsqueeze(0)
        }

    def generate_data(self, me, inp, catidx):
        mgen = MaskedRespGen(self.segsize)
        fmdl = me.narrow_model(catidx, with_softmax=True)
        logging.debug(f"generating {self.nmasks} masks and responses")
        mgen.gen(fmdl, inp, self.nmasks, batch_size=self.batch_size)
        logging.debug("Done generating masks")

        rfactor = inp.numel()
        baseline_score = fmdl(torch.zeros(inp.shape).to(inp.device)).detach().squeeze() * rfactor
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
        )

    def explain(self, me, inp, catidx, data=None, initial=None):

        if data is None:
            data = self.generate_data(me, inp, catidx)
        if initial is None:
            initial = torch.rand(inp.shape[-2:]).to(inp.device)
        sal = optimize_explanation(initial, data.all_masks, data.all_pred, score=data.added_score, 
                                   epochs=self.epochs, lr=self.lr, avg_kernel_size=self.avg_kernel_size,
                                   c_completeness=self.c_completeness, c_smoothness=self.c_smoothness, c_selfness=self.c_selfness,
                                   c_mask_completeness=self.c_mask_completeness)
        
        return sal
