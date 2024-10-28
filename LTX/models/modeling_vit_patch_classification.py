from typing import Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.vit import ViTPreTrainedModel, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.vit.modeling_vit import ViTEncoder
import timm
from config import config

vit_config = config["vit"]
EPSILON = 0.05


class ViTForMaskGeneration(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        proj_shape = self.vit.patch_embed.proj.weight.shape
        self.hidden_size = proj_shape[0]
        self.patch_size = proj_shape[2]        

        self.patch_pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_classifier = nn.Linear(config.hidden_size, 1)  # regression to one number

    @staticmethod
    def from_pretrained(model_name):
        return ViTForMaskGeneration(model_name)
    
    def forward(self, x):
        patch_embeddings = None

        def hook_fn(module, input, output):
            nonlocal patch_embeddings
            patch_embeddings = output

        hook = self.vit.blocks[-1].register_forward_hook(hook_fn)
        try:
            self.vit(x)
        finally:
            hook.remove()

        tokens_output = patch_embeddings[:, 1:, :]
        batch_size = tokens_output.shape[0]
        hidden_size = tokens_output.shape[2]
        
        tokens_output_reshaped = tokens_output.reshape(-1, hidden_size)
        print(tokens_output_reshaped.shape)

        tokens_output_reshaped = self.patch_pooler(tokens_output_reshaped)
        tokens_output_reshaped = self.activation(tokens_output_reshaped)

        logits = self.patch_classifier(tokens_output_reshaped)
        mask = logits.view(batch_size, -1, 1)  # logits - [batch_size, tokens_count]

        mask = mask.view(batch_size,1,int(224/self.patch_size),int(224/self.patch_size))
        interpolated_mask = torch.nn.functional.interpolate(mask, scale_factor=self.patch_size, mode='bilinear')
        return interpolated_mask, mask
        


class ViTForMaskGenerationOrig(ViTPreTrainedModel):
    vit: ViTModel
    patch_classifier: nn.Linear

    def __init__(self, config):
        super().__init__(config)

        # self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)
        # Classifier head
        self.patch_pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_classifier = nn.Linear(config.hidden_size, 1)  # regression to one number
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            pixel_values=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            interpolate_pos_encoding=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs: BaseModelOutputWithPooling = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        tokens_output = sequence_output[:, 1:, :]
        # tokens_output - [batch_size, tokens_count, hidden_size]
        # truncating the hidden states to remove the CLS token, which is the first

        batch_size = tokens_output.shape[0]
        hidden_size = tokens_output.shape[2]

        tokens_output_reshaped = tokens_output.reshape(-1, hidden_size)
        tokens_output_reshaped = self.patch_pooler(tokens_output_reshaped)
        tokens_output_reshaped = self.activation(tokens_output_reshaped)
        # tokens_output_reshaped = self.dropout(tokens_output_reshaped)
        logits = self.patch_classifier(tokens_output_reshaped)
        mask = logits.view(batch_size, -1, 1)  # logits - [batch_size, tokens_count]

        if vit_config["activation_function"] == 'relu':
            mask = torch.relu(mask)
        if vit_config["activation_function"] == 'sigmoid':
            mask = torch.sigmoid(mask)
        if vit_config["activation_function"] == 'softmax':
            mask = torch.softmax(mask, dim=1)

        mask = mask.view(batch_size, 1, int(vit_config["img_size"] / vit_config["patch_size"]),
                         int(vit_config["img_size"] / vit_config["patch_size"]))

        interpolated_mask = torch.nn.functional.interpolate(mask, scale_factor=vit_config["patch_size"],
                                                            mode='bilinear')

        return interpolated_mask, mask
