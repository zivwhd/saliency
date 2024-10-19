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
        model_name = me.arch
        model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
            explainee_model_name=model_name,
            explainer_model_name=model_name,
            activation_function=self.activation_function,
            img_size=inp.shape[-2:],
        )

        logging.info(f"loaded {type(model_for_classification_image)} {type(model_for_mask_generation)} {feature_extractor is None}")
        checkpoint_path = os.path.join(self.checkpoint_base_path, f"{model_name}_{self.variant}.ckpt")
        logging.info("checkpoint_path: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        logging.info("#########################################")
        logging.info(checkpoint['state_dict'].keys())
        logging.info("#########################################")
        model_for_mask_generation.load_state_dict(checkpoint['state_dict'])
        model_for_mask_generation.eval()
        #mask_model = type(model_for_mask_generation).load_from_checkpoint(checkpoint_path)
        mask_model = model_for_mask_generation
        device = inp.device
        with torch.no_grad():
            sal = mask_model.to(inp)
        
        logging.info(f"sal shape: {sal.shape}")
        sssss
        return {"pLTX" : sal.cpu()}
