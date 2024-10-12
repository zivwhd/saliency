from dataclasses import dataclass
import torch
import copy
import logging
import torchvision.transforms as T

@dataclass
class AreaNode:
    def __init__(self, logit, prob, area):
        self.logit = logit
        self.prob = prob
        self.area = area
        self.partitions = []

    def get_area(self):
        return  (self.area[1][0]-self.area[0][0])*(self.area[1][1]-self.area[0][1])

    def __str__(self):
        return f"AreaNode(logit={self.logit}, prob={self.prob}, area={self.area}, [{len(self.partitions)}])"
    
    def __repr__(self):
        return str(self)

class TreGen:

    def __init__(self, me, inp, label, baseline=None, with_blur=True):
        self.me = me
        self.inp = inp
        self.label = label
        self.shape = inp.shape[-2:]
        self.yidx = torch.arange(self.shape[0]).unsqueeze(1).to(inp.device)
        self.xidx = torch.arange(self.shape[1]).unsqueeze(0).to(inp.device)
        if baseline is None:
            self.baseline = torch.zeros(inp.shape).to(inp.device)
        else:
            self.baseline = baseline.to(inp.device)
        self.root = None
        if with_blur:
            self.blur = T.GaussianBlur(kernel_size=17, sigma=2)
        else:
            self.blur = lambda x: x.int()

        self.agg_weights = torch.zeros(self.shape)
        self.agg_resp = torch.zeros(self.shape)



    def split_area(self, area, parts):
            
        base = area[0]
        arm = (area[1]-area[0])/parts
        res = []
        pad = arm*0.4     #torch.tensor([20.0,20.0])
        for yidx in range(parts):
            for xidx in range(parts):
                pos = torch.tensor((yidx*arm[0], xidx*arm[1]))
                topleft = base + pos
                res.append(torch.stack((topleft-pad, topleft + arm +pad)))
                
        return res
    
    def traverse(self, *args, **kargs):
        with torch.no_grad():
            return self.traverse_i(*args, **kargs)
        
    def traverse_i(self, limit = 4, area_limit=16*16):
                
        self.root = self.probe([torch.tensor(((0.0,0.0), self.shape))] )[0]
        pqueue = [self.root]

        idx = 0
        while pqueue and idx <= limit:            
            curr = pqueue.pop(0)
            area = curr.area
            if curr.get_area() < area_limit:
                continue
            idx += 1            
            partitions = []
            for nsplit in [2,3]:
                partitions.append(self.split_area(area, nsplit))

            anodes = self.probe([x  for prt in partitions for x in prt])
            

            curr.partitions = []
            base = 0
            for prt in partitions:
                curr.partitions.append(anodes[base:base+len(prt)])
                base += len(prt)

            pqueue += anodes            
            sal = self.agg_resp  / self.agg_weights        
            def bla(area): 
                pad = 20
                mask = ((self.yidx >= area[0][0]-pad) & (self.yidx < area[1][0]+pad) & (self.xidx >= area[0][1]-pad) & (self.xidx < area[1][1]+pad))
                return (sal * mask.cpu()).mean()
            pqueue.sort(key=lambda x: -bla(x.area))


    def probe(self, area_list, batch_size=32):
        #logging.info(f"probing {area_list}")
        tnodes = []
        yidx = self.yidx
        xidx = self.xidx
        inp = self.inp
        device = inp.device        
        buff = []
        mask_buff = []

        left_area_list = list(area_list)
        for idx, area in enumerate(area_list):
            mask = ((yidx >= area[0][0]) & (yidx < area[1][0]) & (xidx >= area[0][1]) & (xidx < area[1][1]))
            if self.blur:
                mask =  self.blur(mask.unsqueeze(0).float())

            masked_image = mask * inp + (1-mask) * self.baseline
            buff.append(masked_image)
            mask_buff.append(mask.detach())
            if len(buff) >= batch_size or idx == len(area_list)-1:                                
                batch = torch.concat(buff)
                logits = self.me.model(batch).detach()
                probs = torch.softmax(logits, dim=1)
                #print(batch.shape, logits.shape, probs.shape)
                added_tnodes = []
                for aidx in range(len(buff)):
                     
                    added_tnodes.append(AreaNode(
                        logit=logits[aidx, self.label].item(),
                        prob=probs[aidx, self.label].item(),
                        area=left_area_list.pop(0)))                    
                tnodes += added_tnodes

                bmasks = torch.concat(mask_buff)                                        
                areats = torch.tensor([x.get_area() for x in added_tnodes])
                #print(bmasks.shape, areats.shape)
                emasks = (bmasks.cpu() / areats.unsqueeze(1).unsqueeze(1))
                self.agg_weights += emasks.sum(dim=0).cpu()
                self.agg_resp += (emasks * probs[:,self.label].cpu().unsqueeze(1).unsqueeze(1)).sum(dim=0).cpu()
                    
                buff.clear()
                mask_buff.clear()
        assert not buff
        return tnodes
    
    def create_aggsal(self):
        return self.agg_resp  / self.agg_weights        

    def create_sal(self, mode="logit"):
        yidx = torch.arange(self.shape[0]).unsqueeze(1)
        xidx = torch.arange(self.shape[1]).unsqueeze(0)
        sal = torch.zeros(self.shape)

        if mode == "prob":
            valf = lambda x: x.prob
        elif mode == "logit":
            valf = lambda x: x.logit
        elif mode == "pgit":
            valf = lambda x: max(0.00001, x.logit)
        elif mode == "spgit":
            valf = lambda x: x.prob if (x.logit < 0) else (x.logit+0.5)


        todo = [(1.0, self.root)]
        while todo:
            base_score, curr = todo.pop(0)
            if curr.partitions:
                for partition in curr.partitions:
                    cprobs = torch.tensor([valf(x) for x in partition])
                    #careas = torch.tensor([x.get_area() for x in partition])
                    cscores = (base_score * cprobs ) / (cprobs.sum())
                    todo += [(cscores[cidx], child) for cidx, child in enumerate(partition) ]
            else:
                area = curr.area
                mask = ((yidx >= area[0][0]) & (yidx < area[1][0]) & (xidx >= area[0][1]) & (xidx < area[1][1]))
                sal[mask] += base_score / curr.get_area()
        return sal

class TreSaliencyCreator:
    def __init__(self, limit=100):
        self.limit = limit

    def __call__(self, me, inp, catidx):
        res = {}
        tre = TreGen(me, inp, catidx, baseline=None)
        tre.traverse(self.limit)

        modes = ["logit","prob","pgit","spgit"]
        for mode in modes:
            sal = tre.create_sal(mode=mode)
            res[f"TRE_{mode}_{self.limit}"] = sal.unsqueeze(0)
        return res




            

    

    
