import sys, os, logging, time
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import socket, time
from reports import report_duration
def setup_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    targets = [ os.path.join(os.path.dirname(current_dir),"LTX")  ]
    for path in targets:
        if path not in sys.path:
            logging.info(f"adding {path}")
            sys.path.append(path)

import metrics

CHECKPOINT_BASE_PATH = "/home/weziv5/work/products/ltx"


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data  # data is a list of (tensor, target) pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, target =  self.data[idx]
        inp = inp[0].cpu()
        return dict(
            image_name="finetuned_image",
            pixel_values=inp,#None,
            resized_and_normalized_image=inp,
            image=inp,
            target_class=target,
        )
        
def sum_model_weights(model):
    total_sum = sum(param.sum() for param in model.parameters())
    return total_sum.item()        

class LTXSaliencyCreator:
    def __init__(self, activation_function="sigmoid", 
                 variant="vnl", 
                 checkpoint_base_path=CHECKPOINT_BASE_PATH,
                 cp_gen=None):
        setup_path()
        self.activation_function = activation_function       
        self.variant=variant 
        self.checkpoint_base_path = checkpoint_base_path
        self.cp_gen = cp_gen

    def __call__(self, me, inp, catidx):    
        from main.seg_classification.model_types_loading import load_explainer_explaniee_models_and_feature_extractor
        from config import config
        from utils.vit_utils import get_params_from_config, freeze_multitask_model
        from main.seg_classification.image_classification_with_token_classification_model import (
            ImageClassificationWithTokenClassificationModel,
        )        
        from main.seg_classification.image_classification_with_token_classification_model_opt import OptImageClassificationWithTokenClassificationModel
        from main.seg_classification.image_classification_with_token_classification_model import ImageClassificationWithTokenClassificationModel
        #from main.segmentation_eval.segmentation_model import OptImageClassificationWithTokenClassificationModel
        from main.seg_classification.image_token_data_module_opt_segmentation import ImageSegOptDataModuleSegmentation
        
        
        load_start_time = time.time()
        model_name = me.arch
        
        args = get_params_from_config(config_vit=config[model_name])
        logging.info(f"args : {args}")
        img_size = args["img_size"] ##tuple([int(x) for x in inp.shape[-2:]])
        logging.info(f"img_size={img_size}; ")

        model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
            explainee_model_name=model_name,
            explainer_model_name=model_name,
            activation_function=self.activation_function,
            img_size=img_size
        )

        logging.info(f"model weight sum: {sum_model_weights(me.model)}, {sum_model_weights(model_for_classification_image)}")
        warmup_steps = 0
        training_steps = 30
        ins_weight=args["ins_weight_finetune"]
        del_weight=args["del_weight_finetune"]
        prediction_loss_mul = args["prediction_loss_mul_finetune"]
        cp_loss_mul=args["cp_loss_mul"]
        is_convnet = ("vit" not in model_name)

        if self.cp_gen:
            cp_data = self.cp_gen.generate_data(me, inp, catidx)
        else:
            cp_data = None

        model = ImageClassificationWithTokenClassificationModel(
            model_for_classification_image=model_for_classification_image,
            model_for_mask_generation=model_for_mask_generation,
            is_clamp_between_0_to_1=args["is_clamp_between_0_to_1"],
            plot_path=None,##plot_path,
            warmup_steps=warmup_steps, ##warmup_steps,
            total_training_steps=training_steps,##total_training_steps,
            experiment_path=None,##experiment_perturbation_results_path,
            is_explainer_convnet=is_convnet,
            is_explainee_convnet=is_convnet,
            lr=args["lr_finetune"],
            start_epoch_to_evaluate=0,
            n_batches_to_visualize=args["n_batches_to_visualize"],
            mask_loss=args["mask_loss"],
            mask_loss_mul=args["mask_loss_mul"],
            cp_loss_mul=cp_loss_mul,
            prediction_loss_mul=prediction_loss_mul,
            activation_function=args["activation_function"],
            train_model_by_target_gt_class=args["train_model_by_target_gt_class"],
            use_logits_only=args["use_logits_only"],
            img_size=img_size,
            patch_size=args["patch_size"],
            is_ce_neg=args["is_ce_neg"],
            verbose=True, ## args.verbose,            
            is_finetune=True,
            ins_weight=ins_weight,
            del_weight=del_weight,
            cp_data=cp_data
        )

        logging.info(f"loaded {type(model_for_classification_image)} {type(model_for_mask_generation)} {feature_extractor is None}")
        checkpoint_path = os.path.join(self.checkpoint_base_path, f"{model_name}_{self.variant}.ckpt")
        logging.info(f"checkpoint_path: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        #logging.info("#########################################")
        #logging.info(checkpoint['state_dict'].keys())
        #logging.info("#########################################")
        logging.info(f"model weight sum: (1) {sum_model_weights(me.model)}, {sum_model_weights(model_for_classification_image)} {sum_model_weights(model.vit_for_classification_image)}")
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"model weight sum: (2) {sum_model_weights(me.model)}, {sum_model_weights(model_for_classification_image)} {sum_model_weights(model.vit_for_classification_image)}")
        model.eval()
        logging.info(f"model weight sum: (3) {sum_model_weights(me.model)}, {sum_model_weights(model_for_classification_image)} {sum_model_weights(model.vit_for_classification_image)}")
        mask_model = model.vit_for_patch_classification.to(inp.device)
        #mask_model = type(model_for_mask_generation).load_from_checkpoint(checkpoint_path)
        load_end_time = time.time()        
        logging.info(f"load time (sec): {load_end_time-load_start_time}")

        

        model = freeze_multitask_model(
            model=model,
            is_freezing_explaniee_model=args["is_freezing_explaniee_model"],
            explainer_model_n_first_layers_to_freeze=args["explainer_model_n_first_layers_to_freeze"],
            is_explainer_convnet=is_convnet,
        )

        device = inp.device
        with torch.no_grad():
            interpolated_mask, tokens_mask = mask_model(inp)
            logging.info(f"interpolated: {interpolated_mask.shape}; tokens_mask: {tokens_mask.shape}")
            psal = interpolated_mask
        
        idataset = SimpleDataset([(inp, catidx)])
        dl = DataLoader(idataset, batch_size=1)
        data_module = ImageSegOptDataModuleSegmentation(
            train_data_loader=dl
        )

        start_time = time.time()
        trainer = pl.Trainer(
            logger=[],
            accelerator='gpu',
            devices=1,
            #gpus=1,
            #devices=[1, 2],
            num_sanity_val_steps=0,
            #check_val_every_n_epoch=300,
            max_epochs=training_steps, #args.n_epochs_to_optimize_stage_b,
            #resume_from_checkpoint=CKPT_PATH,
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir=args["default_root_dir"],
            #weights_summary=None            
        )

        trainer.fit(model=model, datamodule=data_module)

        logging.info("finetuning")
        

        with torch.no_grad():
            mask_model = model.vit_for_patch_classification.to(inp.device)            
            interpolated_mask, tokens_mask = mask_model(inp)
            logging.info(f"interpolated: {interpolated_mask.shape}; tokens_mask: {tokens_mask.shape}")
            logging.info(f"inp/mask: {inp.sum()}, {interpolated_mask.sum()} {inp.device}")
            
            sal = interpolated_mask

            sel_sal = model.selection[1].to(inp.device)            

            #mt = metrics.Metrics() 
            #model.vit_for_classification_image.to           
            
            #mins = mt.pert_metrics(model_for_classification_image.to(inp.device), inp, sal[0].to(inp.device), catidx, is_neg=True, nsteps=20)            
            #pins = mt.pert_metrics(me.model, inp, sal[0], catidx, is_neg=True, nsteps=20)
            #logging.info(f"{mins} {pins}")
            #logging.info(f"model weight sum: {sum_model_weights(me.model)}, {sum_model_weights(model_for_classification_image)} {sum_model_weights(model.vit_for_classification_image)}")

        
        report_duration(start_time, me.arch, "LTX", training_steps)
        report_duration(load_start_time, me.arch, "LTX_WITH_LOAD", training_steps)
        mask_loss_mul=args["mask_loss_mul"]        
        lr = args["lr_finetune"]
        adesc = f'{mask_loss_mul}_{prediction_loss_mul}_{lr}'
        if cp_data:
            adesc += f'_cp{cp_loss_mul}'
        rv = {
            "pLTX" : psal.cpu()[0], 
            f"LTX_{adesc}" : sal.cpu()[0],
            f"sLTX_{adesc}_{del_weight}_{ins_weight}" : sel_sal.cpu()}

        logging.info(f"sal shapes {[x.shape for x in rv.values()]}")
        return rv
