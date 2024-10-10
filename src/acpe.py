from dataclasses import dataclass
import torch
import copy
import logging

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

    def __init__(self, me, inp, label, baseline=None):
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
    
    def split_area(self, area, parts):
            
        base = area[0]
        arm = (area[1]-area[0])/parts
        res = []
        for yidx in range(parts):
            for xidx in range(parts):
                pos = torch.tensor((yidx*arm[0], xidx*arm[1]))
                topleft = base + pos
                res.append(torch.stack((topleft, topleft + arm )))
                
        return res
    
    def traverse(self, limit = 4, area_limit=16*16):
                
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
            pqueue.sort(key=lambda x: (-x.get_area()*0, -x.logit))


    def probe(self, area_list, batch_size=32):
        logging.info(f"probing {area_list}")
        tnodes = []
        yidx = self.yidx
        xidx = self.xidx
        inp = self.inp
        device = inp.device        
        buff = []
        left_area_list = list(area_list)
        for idx, area in enumerate(area_list):
            mask = ((yidx >= area[0][0]) & (yidx < area[1][0]) & (xidx >= area[0][1]) & (xidx < area[1][1]))
            masked_image = mask * inp + (~mask) * self.baseline          
            buff.append(masked_image)            
            if len(buff) >= batch_size or idx == len(area_list)-1:                                
                batch = torch.concat(buff)
                logits = self.me.model(batch)
                probs = torch.softmax(logits, dim=1)
                #print(batch.shape, logits.shape, probs.shape)
                for aidx in range(len(buff)):                    
                    tnodes.append(AreaNode(
                        logit=logits[aidx, self.label].item(),
                        prob=probs[aidx, self.label].item(),
                        area=left_area_list.pop(0)))                    
                buff.clear()        
        assert not buff
        return tnodes
    
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
            res["TRE_{mode}_{self.limit}"] = sal




            

    

    
