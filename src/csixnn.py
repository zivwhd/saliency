import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import logging, time, os, sys
import pickle
from dataset import CustomImageNetDataset
import functools

BASE_PATH = "products/cpath"

def get_cp_path(base_path, model_name, target):
    return os.path.join(base_path, model_name, f"{target}.cp")

def traverse_module_children(module, path=[], sep='_', rec_list=[], verbose=False):
                
    current_name = sep.join(path)
    if len(path) > 20:
        raise Exception("Bad recursion")
    children = list(module.named_children())
    if verbose:
        print(path, ' --> ', [x[0] for x in children])

    if len(children) == 0:
        return ([current_name], [module])    
    module_list, name_list = [], []
    
    for name, child in children:        
        next_path = path + [name]
        next_name = sep.join(next_path)
        if next_name in rec_list:
            ch_names, ch_modules = traverse_module_children(child, next_path, sep=sep, rec_list=rec_list, verbose=verbose)
        else:
            ch_names, ch_modules = [next_name], [child]
        module_list += ch_modules
        name_list += ch_names
    return name_list, module_list


def simplify_resnet(model):
    REC_LIST = ['layer4', 'layer4_2']
    module_names, module_list = traverse_module_children(model, rec_list=REC_LIST, verbose=False)
    module_names.insert(-1, 'flatten')
    module_list.insert(-1, nn.Flatten(1))
    flat_model = nn.Sequential(OrderedDict(zip(module_names, module_list)))    
    return flat_model


class SimpleResnet50(nn.Module):

    def __init__(self, inner):
        super().__init__()
        ## hide params
        self.inner = [inner]
        layer4_modules = list(inner.layer4.children())
        self.layer4_head = [nn.Sequential(*layer4_modules[0:-1]  )]
        self.layer4_tail = [layer4_modules[-1]]

        self.bt_conv2 = layer4_modules[-1].conv2
        self.bt_bn2 = layer4_modules[-1].bn2
        self.bt_conv3 = layer4_modules[-1].conv3
        self.bt_bn3 = layer4_modules[-1].bn3

        self.avgpool = inner.avgpool
        self.fc = inner.fc
        self.cache = {}
        self.prev_sig = None
        self.prev_vals = None

    def sig(self, x):
        sig =  (
            x.sum().cpu().float().tolist(), 
            (x.flatten().cpu() * torch.arange(x.numel())).sum().float().tolist()
        )
        
        return sig
    
    
    def forward(self, x):
        sig = self.sig(x)
        cvals = self.cache.get(sig)
        if sig == self.prev_sig:
            out, identity = self.prev_vals
        if cvals is not None:
            out, identity = cvals
            out = out.to(x.device)
            identity = identity.to(x.device)
        else:
            #print("calc")
            x = self.inner[0].conv1(x)
            x = self.inner[0].bn1(x)
            x = self.inner[0].relu(x)
            x = self.inner[0].maxpool(x)

            # Forward through residual layers
            x = self.inner[0].layer1(x)
            x = self.inner[0].layer2(x)
            x = self.inner[0].layer3(x)
            x = self.layer4_head[0](x)
            
            #x = self.layer5(x)
            ################
            identity = x
            # Forward through the three convolutional layers
            out = self.layer4_tail[0].conv1(x)
            out = self.layer4_tail[0].bn1(out)
            out = self.inner[0].relu(out)
            #out = self.layer4_tail[0].conv2(out)
            self.cache[sig] = (out.cpu(), identity.cpu())
            self.prev_sig = sig
            self.prev_vals = (out, identity)
            assert len(self.cache) < 200

        out = self.bt_conv2(out)
        out = self.bt_bn2(out)
        out = self.inner[0].relu(out)

        out = self.bt_conv3(out)
        out = self.bt_bn3(out)

        # If downsampling is needed, apply it to match the dimensions

        # Add the skip connection
        out += identity
        out = self.inner[0].relu(out)
        x = out
        ###############
        # Global average pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

class SimpleVGG16(nn.Module):

    def __init__(self, inner):
        super().__init__()
        ## hide params
        self.inner = [inner]
        self.features_head = [inner.features[0:26]]
        self.features_tail = nn.Sequential(*[x for x in inner.features[26:]])
        self.avgpool = inner.avgpool
        self.classifier = inner.classifier
        #self.cache = {}
        self.prev_sig = None
        self.prev_vals = None

    def sig(self, x):
        
        sig =  (
            tuple(x.shape),
            x.sum().cpu().float().tolist(), 
            (x.flatten().cpu() * torch.arange(x.numel())).sum().float().tolist(),
            ((x.flatten().cpu() ** 2) * torch.arange(x.numel())).sum().float().tolist()
        )
        return sig

    def forward(self, x):

        sig = self.sig(x)

        if sig != self.prev_sig:
            x = self.features_head[0](x)
            if x.shape[0] > 1:
                self.prev_sig = sig 
                self.prev_vals = x               
                logging.debug(f"cached {x.shape} {x.numel()}")            
            else:
                self.prev_sig = None
        else:
            #print("load cache")
            x = self.prev_vals

        x = self.features_tail(x)        
        x = self.avgpool(x)        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class SimpleConvnext(nn.Module):
    
    def __init__(self, inner):
        super().__init__()
        self.inner = [inner]        
        self.norm_pre = inner.norm_pre
        self.stages_tail = nn.Sequential(*inner.stages[-1:])
        self.head = inner.head
        self.prev_sig = None
        self.prev_vals = None

    def sig(self, x):
        
        sig =  (
            tuple(x.shape),
            x.sum().cpu().float().tolist(), 
            (x.flatten().cpu() * torch.arange(x.numel())).sum().float().tolist(),
            ((x.flatten().cpu() ** 2) * torch.arange(x.numel())).sum().float().tolist()
        )
        return sig

    def forward(self, x):

        sig = self.sig(x)

        if sig != self.prev_sig:
            x = self.inner[0].stem(x)
            x = self.inner[0].stages[0:-1](x)
            if x.shape[0] > 1:
                self.prev_sig = sig 
                self.prev_vals = x               
                logging.debug(f"cached {x.shape} {x.numel()}")            
            else:
                self.prev_sig = None
        else:
            x = self.prev_vals

        x = self.stages_tail(x)
        x = self.norm_pre(x)
        x = self.head(x)
        return x
        

def get_simplified_model(me):
    if me.arch == 'resnet50':
        return SimpleResnet50(me.model)
    elif me.arch == 'vgg16':
        return SimpleVGG16(me.model)    
    elif me.arch == 'convnext_base':
        return SimpleConvnext(me.model)    
    
    else:
        raise Exception(f"unexpected arch {me.arch}")

def get_simplified_model_layer_name(me):
    if me.arch == 'resnet50':
        return "bt_conv3"
    elif me.arch == 'vgg16':
        return "features_tail.2"    
    elif me.arch == 'convnext_base':
        return "stages_tail.0.blocks.2.conv_dw" ## 5,6 stages_tail.0.blocks.2.conv_dw
    else:
        raise Exception(f"unexpected arch {me.arch}")
    
def generate_causal_path(me, target, isrc, device='cuda', base_path=BASE_PATH, validate=True):
    from baselines.ixnn import setup
    from explainnn.explain import ExplainNN

    fmdl = get_simplified_model(me)
    
    all_images = list(isrc.get_all_images().values())
    dset = CustomImageNetDataset(all_images)
    args = dict(
        device=device, 
        verbose=True,
        dataset_name="imagenet",
        dataset = dset,
        model_name=me.arch,
        target_idx=target,
        n_samples=100,
        soft_interventions=True,
        graph_stab=True,
        gen_attr=False,
        save_attr=False,
        vis_attr=False,
        eval_attr=False,
        baseline_attr=False,
    )

    if me.arch in ['vgg16']:
        args["override_layers_index"] = [3,4] ## or 3,4
        logging.info("VGG16 override")
    elif me.arch in ['convnext_base']:
        args["override_layers_index"] = [5,6] ## or 3,4
        logging.info("ConvNext override")

    
    explainer = ExplainNN(fmdl, args, beta=0, device=me.device, verbose=True)
    logging.info("generating causal path")
    start_time = time.time()
    causal_path = explainer.run()
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"DONE generating causal path {duration}")

    cpath = get_cp_path(base_path, me.arch, target)    
    logging.info(f'saving {cpath} {causal_path.keys()}')
    if validate:
        expected_layer = get_simplified_model_layer_name(me)
        logging.info(f"expected layer: {expected_layer}")
        assert expected_layer in causal_path.keys()
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    
    with open(cpath,"wb") as cpf:
        pickle.dump(causal_path, cpf)

    

class IXNNSaliencyCreator:
    def __init__(self, base_path=BASE_PATH):
        self.base_path = base_path

    def __call__(self, me, inp, catidx):
        from baselines.ixnn import setup
        from explainnn.explain import ExplainNN

        causal_path = self.get_causal_path(me.arch, catidx)

        logging.info(f"causal_path: {causal_path.keys()}")
        args = dict(
            device=inp.device, 
            verbose=True,
            #dataset_name="imagenet",
            #dataset = dset,
            model_name=me.arch,
            target_idx=catidx,
            n_samples=100,
            soft_interventions=True,
            graph_stab=True,
            gen_attr=False,
            save_attr=False,
            vis_attr=False,
            eval_attr=False,
            baseline_attr=False,
        )
            
        fmdl = get_simplified_model(me)
        fmdl_layer_name = get_simplified_model_layer_name(me)
        explainer = ExplainNN(fmdl, args, beta=0, device=me.device, verbose=True)
        attributions = explainer.extract_attributions(inp, fmdl_layer_name, causal_path)

        vis = {}
        att = attributions['attributions']
        features  = np.asarray(att)
        for i in range(features.shape[0]):
            feat = features[i]
            vmin, vmax = np.min(feat), np.max(feat)
            feat = (feat - vmin) / (vmax - vmin)

            feat = 2 * feat - 1 
            feat = feat[:, :, np.newaxis]
            vis[attributions['indices'][i]] = feat

        sal = np.zeros((224,224,1))
        for kk, ft in vis.items():
            sal += ft
        
        sal = torch.tensor(sal).squeeze(2).unsqueeze(0).float()
        logging.debug(f"sal-shape {sal.shape}")
        return dict(IXNN=sal)

    @functools.lru_cache(maxsize=None)
    def get_causal_path(self, arch, catidx):
        cpath = get_cp_path(self.base_path, arch, catidx)
        
        with open(cpath,"rb") as cpf:
            cp = pickle.load(cpf)
            logging.info(f"loaded causal path at {cpath}: {cp.keys()}")
            return cp
 

