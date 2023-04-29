from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule
from ..utils import hf_get_causal_hidden_layers

class RewardModel(LoRAModule):
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 model: nn.Module,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 freeze_layer_ratio: float = 0.5) -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        hidden_layers = hf_get_causal_hidden_layers(model)
        num_layers_freeze = int(len(hidden_layers) * freeze_layer_ratio)
        if num_layers_freeze > 0:
            hidden_layers_to_freeze = list(hidden_layers)[:num_layers_freeze]
        else:
            hidden_layers_to_freeze = []

        for layer in hidden_layers_to_freeze:
            layer.requires_grad_(False)

        self.convert_to_lora()

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        value = outputs['logits']
        return value
