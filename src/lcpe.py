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
                beta=0.1, alpha=0.0, 
                avg_kernel_size=(17,17),
                ):
    criterion = nn.MSELoss()  # Mean Squared Error loss
    #print(list(model.parameters()))
    logging.debug(f"### lr={lr}; alpha={alpha}; beta={beta};")
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

        if alpha:
            explanation_sum = model.explanation.sum()
            explanation_loss = criterion(explanation_sum/exp.numel(), score/ exp.numel()) 
        else:
            explanation_loss = 0

        
        conv_loss = 0
        if beta:                
            sexp = F.conv2d(exp.unsqueeze(0), avg_kernel.unsqueeze(0),padding="same").squeeze()
            conv_loss = criterion(exp, sexp)            
        else:
            conv_loss = 0

        # Backward pass and optimization
        explanation_loss = 0
        total_loss = comp_loss + beta * conv_loss + alpha * explanation_loss
        
        total_loss.backward()        
        optimizer.step()

        # Normalize the explanation with the given score
        #model.normalize(score)
        
        logging.debug(f"Epoch {epoch+1}/{epochs} Loss={total_loss.item()}; ES={model.explanation.sum()}; comp_loss={comp_loss}; exp_loss={explanation_loss}; conv_loss={conv_loss}")


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

class CompExpCreator:

    def __init__(self, nmasks=500, segsize=48, batch_size=32, 
                 lr = 0.05, alpha=0, beta=1.0, avg_kernel_size=(17,17),
                 epochs=500,
                 **kwargs):
        self.segsize = segsize
        self.nmasks = nmasks
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta  
        self.lr = lr
        self.avg_kernel_size = avg_kernel_size      
        self.epochs = epochs

    def __call__(self, me, inp, catidx):
        sal = self.explain(me, inp, catidx)
        ksdesc = str("x").join(map(str, self.avg_kernel_size))
        return {
            f"Comp_{self.nmasks}_{self.segsize}_{ksdesc}" : sal.cpu().unsqueeze(0)
        }

    def explain(self, me, inp, catidx):

        mgen = MaskedRespGen(self.segsize)
        fmdl = me.narrow_model(catidx, with_softmax=True)
        logging.debug(f"generating {self.nmasks} masks and responses")
        mgen.gen(fmdl, inp, self.nmasks, batch_size=self.batch_size)
        logging.debug("Done generating masks")

        rfactor = inp.numel()
        baseline_score = fmdl(torch.zeros(inp.shape).to(inp.device)).detach().squeeze() * rfactor
        label_score = fmdl(inp).detach().squeeze() * rfactor

        delta_score = label_score - baseline_score
        #print(label_score, baseline_score, delta_score)
    
        device = me.device
        all_masks = torch.concat(mgen.all_masks).to(device)
        all_targets = torch.concat(mgen.all_pred).to(device).squeeze() * rfactor - baseline_score
        all_targets.shape, baseline_score.shape

        sal = optimize_explanation(torch.rand(inp.shape[-2:]).to(device), all_masks, all_targets, score=delta_score, 
                                   epochs=self.epochs, lr=self.lr)
        
        return sal

