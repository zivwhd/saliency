import sys, os, logging
import torch
def setup_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    targets = [ os.path.join(os.path.dirname(current_dir),"LTX")  ]
    for path in targets:
        if path not in sys.path:
            logging.info(f"adding {path}")
            sys.path.append(path)

    
CHECKPOINT_BASE_PATH = "/home/weziv5/work/products/ltx"
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
        
        model_name = me.arch
        
        args = get_params_from_config(config_vit=config[model_name])
        img_size = tuple([int(x) for x in inp.shape[-2:]])
        logging.info(f"img_size={img_size}; {type(img_size[0])}")

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
            total_training_steps=0,##total_training_steps,
            experiment_path=None,##experiment_perturbation_results_path,
            is_explainer_convnet=is_convnet,
            is_explainee_convnet=is_convnet,
            lr=args["lr"],
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
        
        device = inp.device
        with torch.no_grad():
            sal = mask_model(inp)
        
        logging.info(f"sal shape: {sal.shape}")
        sssss
        return {"pLTX" : sal.cpu()}
