import sys, os, logging, time
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

def setup_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    targets = [ os.path.join(os.path.dirname(current_dir),"LTX")  ]
    for path in targets:
        if path not in sys.path:
            logging.info(f"adding {path}")
            sys.path.append(path)



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
        
        

class LTXSaliencyCreator:
    def __init__(self, activation_function="sigmoid", variant="vnl", checkpoint_base_path=CHECKPOINT_BASE_PATH):
        setup_path()
        self.activation_function = activation_function       
        self.variant=variant 
        self.checkpoint_base_path = checkpoint_base_path

    def __call__(self, me, inp, catidx):    
        from main.seg_classification.model_types_loading import load_explainer_explaniee_models_and_feature_extractor
        from config import config
        from utils.vit_utils import get_params_from_config
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

        is_convnet = ("vit" not in model_name)

        model = ImageClassificationWithTokenClassificationModel(
            model_for_classification_image=model_for_classification_image,
            model_for_mask_generation=model_for_mask_generation,
            is_clamp_between_0_to_1=args["is_clamp_between_0_to_1"],
            plot_path=None,##plot_path,
            warmup_steps=0, ##warmup_steps,
            total_training_steps=30,##total_training_steps,
            experiment_path=None,##experiment_perturbation_results_path,
            is_explainer_convnet=is_convnet,
            is_explainee_convnet=is_convnet,
            lr=args["lr_finetune"],
            start_epoch_to_evaluate=args["start_epoch_to_evaluate"],
            n_batches_to_visualize=args["n_batches_to_visualize"],
            mask_loss=args["mask_loss"],
            mask_loss_mul=args["mask_loss_mul"],
            prediction_loss_mul=args["prediction_loss_mul"],
            activation_function=args["activation_function"],
            train_model_by_target_gt_class=args["train_model_by_target_gt_class"],
            use_logits_only=args["use_logits_only"],
            img_size=img_size,
            patch_size=args["patch_size"],
            is_ce_neg=args["is_ce_neg"],
            verbose=True, ## args.verbose,
        )

        logging.info(f"loaded {type(model_for_classification_image)} {type(model_for_mask_generation)} {feature_extractor is None}")
        checkpoint_path = os.path.join(self.checkpoint_base_path, f"{model_name}_{self.variant}.ckpt")
        logging.info(f"checkpoint_path: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        #logging.info("#########################################")
        #logging.info(checkpoint['state_dict'].keys())
        #logging.info("#########################################")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        mask_model = model.vit_for_patch_classification.to(inp.device)
        #mask_model = type(model_for_mask_generation).load_from_checkpoint(checkpoint_path)
        load_end_time = time.time()        
        logging.info(f"load time (sec): {load_end_time-load_start_time}")
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

        trainer = pl.Trainer(
            logger=[],
            accelerator='gpu',
            devices=1,
            #gpus=1,
            #devices=[1, 2],
            num_sanity_val_steps=0,
            #check_val_every_n_epoch=300,
            max_epochs=30, #args.n_epochs_to_optimize_stage_b,
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
            sal = interpolated_mask

        return {"pLTX" : psal.cpu()[0], "LTX" : sal.cpu()[0]}
