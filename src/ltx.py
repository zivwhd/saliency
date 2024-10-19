import sys, os, logging

def setup_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    targets = [ os.path.join(os.path.dirname(current_dir),"LTX")  ]
    for path in targets:
        if path not in sys.path:
            logging.info(f"adding {path}")
            sys.path.append(path)

    
CHECKPOINT_BASE_PATH = "/home/weziv5/work/products/ltx"
class LTXSaliencyCreator:
    def __init__(self, checkpoint_path, activation_function="sigmoid", variant="vnl"):
        setup_path()
        self.activation_function = activation_function       
        self.variant=variant 

    def __call__(self, me, inp, catidx):    
        from main.seg_classification.model_types_loading import load_explainer_explaniee_models_and_feature_extractor
        model_name = me.arch
        model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
            explainee_model_name=model_name,
            explainer_model_name=model_name,
            activation_function=self.activation_function,
            img_size=inp.shape[-2:],
        )

        checkpoint_path = os.path.join(CHECKPOINT_BASE_PATH, f"{model_name}_{self.variant}.ckpt")
        mask_model = model_for_mask_generation.load_from_checkpoint(checkpoint_path)
        device = inp.device
        sal = mask_model.to(inp)
        
        logging.info(f"sal shape: {sal.shape}")
        return {"pLTX" : sal}
