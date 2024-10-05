import time, pdb

from collections import defaultdict
from scipy.stats import entropy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from explainnn.explain_utils import *
from explainnn.dataset import get_dataset
from explainnn.definitions import OUTPUT_DIR, ATTR_DATASET
import logging

def desc(x):
    if type(x) == dict:
        return f"dict({len(x)})"
    if type(x) == list:
        return f"list({len(x)})"    
    return str(type(x))

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class CachedModelEval:

    def __init__(self):
        self.last_call = None

    def get_structural_module(self, name, model):
        s = name.split('.')
        if len(s)>1:
            module = model._modules[s[0]]
            for idx in range(1, len(s)):
                module = module._modules[s[idx]]
            return module, s[idx]
        else:
            return model._modules[name], name

    def cached_eval(self, model, name, input):
        call_key = (model, layer, input)
        if call_key == self.last_call:
            pass

        module = self.get_structural_module(name, model)
        


class GetALLLayerInformation:

    def __init__(self, model, depth=None, top_down=True, device=torch.device('cpu')):
        super(GetALLLayerInformation, self).__init__()
        self.model = model
        self.handlers = []
        self.actmap = {}
        self.top_down = top_down
        self.depth = depth
        self.Ln_Ln_1_params = dict()
        self.Ln_Ln_1_biases = dict()
        self.act_neurons = dict()
        self.device = device

    def get_global_info(self):
        modules_names = [key.split('.weight')[0] for key, _ in self.model.named_parameters() if 'weight' in key]
        
        if self.top_down:
            reversed_modules_names = [modules_names[len(modules_names)-1-k] for k in range(len(modules_names))]
            layer_name_map = dict(zip(reversed(modules_names), range(len(modules_names))))
            modules_names = reversed_modules_names
        else:
            layer_name_map = dict(zip(modules_names, range(len(modules_names))))
        
        count = 0
        for name in modules_names:
            count += 1
            module, _ = get_structural_module(name, self.model)    
            self.Ln_Ln_1_params[layer_name_map[name]] = list(module.weight.data)
            # check if bias exists
            if not module.bias is None:
                self.Ln_Ln_1_biases[layer_name_map[name]] = list(module.bias.data)
        
        self.feat_dims = get_last_conv_dims(
            self.model, (3,224,224), ## PUSH_ASSERT
            self.device
            )  
        return {'layer_wise_params': self.Ln_Ln_1_params,
                'layer_wise_biases': self.Ln_Ln_1_biases,
                'depth': count,
                'layers_index': self.Ln_Ln_1_params.keys(),
                'layers_names': layer_name_map,
                'pre-flattened_dims': self.feat_dims['pre-flattened']}

    def get_local_info(self, input):

        selected_activation_layers = []
        modules = get_module(self.model)
        if self.top_down:
            modules_names = reversed(modules.keys())
        else:
            modules_names = modules.keys()
        
        for name in modules_names:
            module = modules[name]
            if isinstance(module, torch.nn.BatchNorm2d):
                continue
            self.handlers.append(Hook(module))
            selected_activation_layers.append(name)    
        
        selected_activation_layers.append('input')
        logits = self.model(input)
        probs = F.softmax(logits, dim=1)
        
        self.act_neurons['labels'] = logits
        
        act_dims = []
        
        for k in range(len(self.handlers)):
            self.act_neurons[selected_activation_layers[k]] = self.handlers[k].output
            act_dims.append(self.handlers[k].output[0].squeeze().detach().cpu().numpy().shape)  
        
        self.act_neurons['input'] = self.handlers[-1].input
        act_dims.append(self.handlers[-1].input[0].squeeze().detach().cpu().numpy().shape)
        
        for h in self.handlers:
            h.close()   
        self.handlers = []

        return {'layer_wise_act': self.act_neurons, 
                'input': input, 
                'labels': probs, 
                'y_hat': logits,
                'act_dims': act_dims,
                'selected_activation_layers': selected_activation_layers,
                }

class ExplainNN(GetALLLayerInformation):
    """
    explain the behviour of pretrained neural network
    """
    def __init__(self, model, args, beta, verbose=False, device=torch.device("cpu")):
        super(ExplainNN, self).__init__(model, device=device)
        self.model = model
        self.params = args
        self.output = OUTPUT_DIR
        self.number_samples = args['n_samples']
        self.args = args
        self.target_idx = args['target_idx']
        self.beta = beta        
        self.soft = args['soft_interventions']
        self.do_mediation_analysis = False
        self.verbose = verbose
        self.override_layers_index = args.get("override_layers_index")
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.global_info_dict = self.get_global_info()
    
    def run(self, soft=False):
        ## PPP        
        causal_path = self.cross_sample_path_effect_analysis(self.global_info_dict['layers_index'], soft=soft)
        return causal_path

    def load_input(self, shuffle):
        """
        load a test sample
        """
        val_set = self.params.get('dataset')
        if not val_set:
            _, val_set= get_dataset(self.params['dataset_name'])
        else:
            print("provided dataset", val_set)
        
        print(val_set)
        loader = DataLoader(dataset=val_set, batch_size=1, shuffle=shuffle, num_workers=1)
        
        target_indices = torch.where(torch.as_tensor(loader.dataset.targets) == self.target_idx)[0]
        print(f"num samples for target {self.target_idx}: {target_indices.shape}")    
        logging.debug(f"num samples for target {self.target_idx}: {target_indices.shape}")
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_indices)
        val_loader = DataLoader(dataset=val_set, sampler=sampler, batch_size=1, shuffle=shuffle, num_workers=1) ## PUSH_ASSERT

        x_c = []
        y_c = self.target_idx
        
        
        cs_sample = defaultdict(list)
        l, p  = [], []
        c=0
        logging.debug("loading input")
        for x_in, _ in val_loader:
            x_in = x_in.to(self.device)
            logits = self.model(x_in)
            probs = F.softmax(logits, dim=1).squeeze(0)
            logits = logits.squeeze(0)
            # take postive examples : correct decisions
            if torch.argmax(probs) == y_c:
                
                x_c.append(x_in)
                l.append(logits[y_c])
                p.append(probs[y_c])
                
                if c == self.number_samples:
                    break
                cs_sample['sample'].append(x_in)
                cs_sample['logits'].append(logits)
                cs_sample['probs'].append(probs)
                cs_sample['feat_dims'] = self.feat_dims
                cs_sample['label'] = y_c
                c += 1

        self.number_samples = c
        cs_sample['reference_mean'] = (torch.stack(l).mean().item(), torch.stack(p).mean().item())
        cs_sample['reference_std'] = (torch.stack(l).std().item(), torch.stack(p).std().item())
         
        return cs_sample

    def read_input_dict(self, input_dict):
        self.x_c = input_dict['sample']
        self.y_c = input_dict['label']
        self.observed_prob = torch.stack(input_dict['probs'])
        self.observed_yhat = torch.stack(input_dict['logits'])
        self.pre_flattened_dims = self.feat_dims['pre-flattened']
        self.reference_std = input_dict['reference_std']
        self.reference_mean = input_dict['reference_mean']

    # learn causal structure over N samples of given token/query 
    def cross_sample_path_effect_analysis(self, layers_index, soft=False):
        
        ## PPP
        logging.debug("cross_sample_path_effect_analysis")

        input_dict = self.load_input(shuffle=False)
        self.read_input_dict(input_dict)
        
        effect_neurons = [self.target_idx]
        causal_path = defaultdict()

        if self.override_layers_index:
            logging.info("overriding layers index: {self.override_layers_index}")
            layers_index = self.override_layers_index
            self.get_layer_weights(layers_index[0])
            effect_neurons = list(range(len(self.layer_wise_params)))
            logging.info("setting effect_neurons: {len(effect_neurons)}")
                
        for idx in layers_index:            
            start_time = time.time()
            logging.debug(f"iteration {idx}")
            if self.verbose:
                print(f'layer idx: {idx} / {len(layers_index)} ')
            self.get_layer_weights(idx)
            logging.debug(f"cs layer {idx} {self.L_n_name}")
            if idx == list(layers_index)[-1]:
                # edge case input layer -> first layer
                causal_path[self.L_n_name] = ([0], [], positive_cause)
                print("first-layer", self.L_n_name)
                break
            u=0.0
            if self.soft:
                u = torch.FloatTensor(1).uniform_(self.beta, self.beta+0.01).item()
            logging.debug("calling compute_path_total_effect")
            scores, path_total_effect, Yw, ids = self.compute_path_total_effect(effect_neurons, n_samples=self.number_samples, u=u)
            print(">> scores, total_effect, Yw: ", desc(scores), desc(path_total_effect), desc(Yw))
            logging.debug(f">> scores, total_effect, Yw:  {desc(scores)}, {desc(path_total_effect)}, {desc(Yw)}")
            if Yw is None: 
                logging.info("No Yw")
                continue   
            logging.debug("selecting causal path")
            _, positive_cause = self.select_causal_path(ids, scores, path_total_effect['relative_diff'], 
                                                Yw, observed_Y=self.observed_yhat[:, self.y_c], 
                                                control_stats= (self.reference_mean, self.reference_std), 
                                                thr=0.05, mode='threshold')
            logging.debug(f"done selecting {len(positive_cause)}")
            print(">> positive cause: ", positive_cause)
            if len(positive_cause) == 0:
                print("cannot explore deeper")
                #print("PUSH_ASSERT - continuing instead of breaking")
                break
            causal_path[self.L_n_name]= (positive_cause, effect_neurons)
                
            # update effects
            print("Updating effect_neurons", positive_cause)            
            effect_neurons = positive_cause
            if self.L_n_name == 'classifier.0':
                logging.info("WARN: patch for VGG16 avgpool - mapping indexes")
                effect_neurons = list(set([(idx // 49) for idx in effect_neurons]))
            logging.debug(f">> updating effect neurons {effect_neurons}")
    
            if self.verbose:     
                print("--- %s seconds for algorithm on one layer ---" % (time.time() - start_time))

        if self.verbose:
            print("causal_path:", causal_path)
        causal_path['label'] = self.y_c
        logging.debug("done select causal_path")
        ##  save_causal_graph(causal_path, self.args)    
        return causal_path

    def cross_sample_path_effect_analysis_layer(self, layers_index, causal_graph_in, layer_name, soft=False):
        logging.debug("cross_sample_path_effect_analysis_layer")        
        input_dict = self.load_input(shuffle=False)
        self.read_input_dict(input_dict)
        if self.verbose:
            print(causal_graph_in.keys())
        effect_neurons = [self.target_idx]
        causal_path = defaultdict()
        
        logging.debug("starting iterations")        
        for idx in layers_index:
            self.get_layer_weights(idx)
            logging.debug(f"iteration {idx} : {self.L_n_name} ")        
            if self.L_n_name not in causal_graph_in.keys() and self.L_n_name != layer_name:
                if self.verbose:
                    print(f'Skipping layer {self.L_n_name}, as it is not relevant')
            elif self.L_n_name != layer_name:
                if self.verbose:
                    print(f'Skipping layer {self.L_n_name}, using causal graph instead')
                causal_path[self.L_n_name] = causal_graph_in[self.L_n_name]
            else:
                start_time = time.time()
                if self.verbose:
                    print(f'layer idx : {idx}')
                    print(f'layer name : {self.L_n_name}')
                if idx == list(layers_index)[-1]:
                    causal_path[self.L_n_name] = ([0], [], positive_cause)
                
                elif self.soft:
                    u = np.random.uniform(0, self.beta)
                    scores, path_total_effect, Yw, ids = self.compute_path_total_effect(effect_neurons, n_samples=self.number_samples, u=u)
                _, positive_cause = self.select_causal_path(ids, scores, path_total_effect['relative_diff'], 
                                                Yw, observed_Y=self.observed_yhat[:, self.y_c], 
                                                control_stats= (self.reference_mean, self.reference_std), 
                                                thr=0.05, mode='threshold')
                print("##2##", len(positive_cause))
                if len(positive_cause) == 0:
                    print("cannot explore deeper")
                    break
                causal_path[self.L_n_name]= (positive_cause, effect_neurons)
                    
                # update effects
                effect_neurons = list(positive_cause)
                if self.verbose:
                    print(effect_neurons)
            
                    print("--- %s seconds for algorithm on one layer ---" % (time.time() - start_time))
        causal_path['label'] = self.y_c
        return causal_path

        
    ## returns scores, path_total_effect, Yw, ids
    def compute_path_total_effect(self, neuron_idx, n_samples=1, u=0.0):
        ## neuron_idx here is the target 
        print("comp_total_effect", neuron_idx)
        logging.debug(f"compute_path_total_effect {len(neuron_idx)} {self.L_n_1_name} {self.L_n_name}")
        #logging.debug(f">> {neuron_idx}")
        #self.model, self.L_n_name

        #check edge case: weights are scalar values or 1d vector
        if len(torch.stack(self.layer_wise_params).shape) <= 1 ##: or torch.stack(self.layer_wise_params).shape[1] == 1: - groups 
            logging.debug(f"no layer_wise_params {torch.stack(self.layer_wise_params).shape}")
            scores = [0.0]
            return scores, None, None, None
        else:
            I_weights, flatten_alpha, I_indices = get_interventional_weights(self.layer_wise_params, self.L_n_1_name, 
                                                                 self.L_n_name, neuron_idx, self.pre_flattened_dims)        
        self.reshape_weights = flatten_alpha
        
        score = defaultdict(list)
        Te = defaultdict(list)
        Te_n = defaultdict(list)
        interventional_Y = defaultdict(list)
        ids = {}

        indices = range(0, I_weights.shape[1])
        if len(I_indices) != 0:
            logging.debug(f"with selection {len(I_indices)}")
            selected_indices = I_indices
        else: 
            logging.debug(f"no selection")
            selected_indices = indices  
        
        t = 0        
        print(f"## {len(indices)} {len(selected_indices)} {n_samples} {neuron_idx}")
        logging.debug(f"## {self.L_n_name} indices={len(indices)};  selected={len(selected_indices)}; samples={n_samples}; effect={len(neuron_idx)}")
        import time
        last_time = time.time()
        processed_nidx = 0
        top_start_time = time.time()
        batch_size = 25
        

        for from_sidx in range(0, n_samples, batch_size):                
            to_sidx = min(from_sidx + batch_size, n_samples)
            x = torch.concat([self.x_c[i].to(self.device) for i in range(from_sidx, to_sidx)])

            c = 0
            for itr, idx in enumerate(indices):
                if idx in selected_indices:
                    processed_nidx += 1
                    print(".", end="")
                    if (processed_nidx % 40 == 0):
                        if (processed_nidx % 120 == 0):
                            logging.debug(f"...{processed_nidx}...")
                        print(int(time.time()-last_time), processed_nidx)
                        #processed_nidx = 0
                        last_time = time.time()

                    net = self.model
                    start_time = time.time()

                    ############
                    
                    Yw, Pw, prev_module = multiple_interventions_effect(idx, x, neuron_idx, self.reshape_weights, net, self.L_n_name, I_weights, u=u)
                    ##Yw, Pw (b,1000,) (b,1000,)
                    ## neuron_idx here - is the target we're observing                    
                    for i in range(from_sidx, to_sidx):
                        ti = i - from_sidx
                        entr1 = entropy(Pw[ti], self.observed_prob[i].detach().cpu().numpy())
                        score[idx].append(entr1)
                        Te[idx].append(Yw[ti][self.y_c] - self.observed_yhat[i][self.y_c])  
                        # normalized effect: 
                        Te_n[idx].append((Yw[ti][self.y_c] / self.observed_yhat[i][self.y_c] - 1))
                        interventional_Y[idx].append(Yw[ti][self.y_c])

                    ids[c] = idx
                    c += 1
                    t += (time.time() - start_time)

        print("")
        top_end_time = time.time()
        if self.verbose:
            print("--- %s seconds for all interventions and effects ---" % str(float(t)/(itr+1)))   
            print("--- top time", top_end_time - top_start_time)
        path_total_effect = {'diff': Te, 'relative_diff': Te_n}
        logging.debug("done computing total effect")
        return score, path_total_effect, interventional_Y, ids

    def get_structural_module(self, name, model):
        s = name.split('.')
        if len(s)>1:
            module = model._modules[s[0]]
            for idx in range(1, len(s)):
                module = module._modules[s[idx]]
            return module, s[idx]
        else:
            return model._modules[name], name

    def get_layer_name_by_idx(self, layer_index):
        layer_name = list(self.global_info_dict['layers_names'].keys())[layer_index]
        return layer_name

    def get_layer_weights(self, layer_index):
        
        self.layer_indices = self.global_info_dict['layers_index']
        self.layer_names = self.global_info_dict['layers_names']
        
        desc = {idx : len(self.global_info_dict['layer_wise_params'][idx]) for idx in list(self.layer_indices)[0:10]}
        logging.debug(f"$$$ {layer_index} {desc}")

            
        if self.top_down:
            self.L_n_name = list(self.global_info_dict['layers_names'].keys())[layer_index] 
            if self.verbose:
                print("explain NN through path-specific effects - path analysis approach ::: ", self.L_n_name)

            if layer_index == list(self.layer_indices)[-1]:
                self.L_n_1_name = 'input'
            else:
                self.L_n_1_name = list(self.global_info_dict['layers_names'].keys())[layer_index+1]
            
            self.layer_wise_params = self.global_info_dict['layer_wise_params'][layer_index]
        else:
           raise ValueError("path-specific effect is a top-down appraoch, check that top_down = True")

    def select_causal_path(self, ids, scores, total_effect, Yw, observed_Y, control_stats, thr=0.05, mode='top_n'):
        
        param_indices = torch.tensor(list(Yw.keys()))
        Yw = torch.tensor(list(Yw.values()))
        total_effect = torch.tensor(list(total_effect.values()))
        scores = torch.tensor(list(scores.values()))
        get_topk = False

        def topK(te, K=3):
            te = te.squeeze()
            return torch.argsort(te, dim=0)[:K]
        
        if mode == 'top_n':
            topK_positive_effect_idx = topK(total_effect)
            topK_score_idx = topK(-scores)

            topK_positive_effect_idx = param_indices[topK_positive_effect_idx]
            topK_score_idx = param_indices[topK_score_idx]
            return topK_score_idx, topK_positive_effect_idx
        
        if mode == 'threshold':
            #pdb.set_trace()
            g1_mean = Yw.mean(1)
            g1_std = Yw.std(1)
            g2_mean = control_stats[0][0]
            g2_std = control_stats[1][0]
            print("Yw.shape", Yw.shape)
            if Yw.shape[1] == 1:
                print(">> minimal effect", total_effect.squeeze().min() )
                positive_effect_idx = torch.tensor(torch.where(total_effect.squeeze() < thr)).squeeze()
                positive_effect_idx = param_indices[positive_effect_idx]
                
            else:
                
                positive_effect_idx = []
                
                #method1 z score : difference of means 
                var_of_diff = torch.sqrt(g1_std**2/(self.number_samples) + g2_std**2/self.number_samples)
                dist = var_of_diff * 1.65 # z-score at pvalue 0.05 test
                #print(" >> minimal_effect dist",  dist)
                #print("   ", g1_mean)
                #print("   ", g2_mean)
                positive_effect_idx = torch.where(g1_mean < g2_mean - dist)[0]
                positive_effect_idx = param_indices[positive_effect_idx]

        # we can choose top K scores then        
        if get_topk and len(positive_effect_idx) > 0:
            K = int(0.8 * len(positive_effect_idx))
            topk = topK(total_effect.mean(1), K=K)
            topk = [ids[k.item()] for k in topk]
            positive_effect_idx = list(topk)
    
        elif len(positive_effect_idx) == 0: 
            print("WARNING: positive_effect_idx is 0.0 after thresholding, altenative method is used instead...")            
            te = total_effect.mean(1)
            print(">> te:", te.min(), te.max())
            p_idx = torch.where(te < -0.001)[0]
            if p_idx.numel() == 0:
                print("WARNING: none selected")
                p_idx = torch.where(te <= te.quantile(0.05))[0]
            positive_effect_idx = [ids[k.item()] for k in p_idx]
            print(">> selected", len(positive_effect_idx))
            logging.debug(f">> selected {len(positive_effect_idx)} out of {len(ids)}")

        return [], positive_effect_idx

    def extract_attributions(self, x, layer_name, causal_path):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        
        x = x.to(self.device)
        out_dict = self.get_local_info(x)
        
        layer_activations = out_dict['layer_wise_act']
        weights = self.global_info_dict['layer_wise_params']
        layer_maps = self.global_info_dict['layers_names']
        
        attributions = get_path_specific_visual_attributions(x, layer_activations, weights, causal_path, layer_name, layer_maps)    
        return attributions