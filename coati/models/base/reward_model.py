from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule


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
                 lora_train_bias: str = 'none') -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

        # if value_head is not None:
        #     if value_head.out_features != 1:
        #         raise ValueError("The value head of reward model's output dim should be 1!")
        #     self.value_head = value_head
        # else:
        #     self.value_head = nn.Linear(model.config.n_embd, 1)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        value = outputs['logits']
        return value
        
        # last_hidden_states = outputs['last_hidden_state']
        # values = self.value_head(last_hidden_states)[:, :-1]
        # value = values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        # return value
