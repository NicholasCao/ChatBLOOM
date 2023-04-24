from typing import Optional

import torch.nn as nn
from transformers import BloomConfig, BloomForSequenceClassification

from ..base import RewardModel


class BLOOMRM(RewardModel):
    """
    BLOOM Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 pretrained: str = None,
                 config: Optional[BloomConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = BloomForSequenceClassification.from_pretrained(pretrained, num_labels=1)
        elif config is not None:
            config.num_labels = 1
            model = BloomForSequenceClassification(config)
        else:
            config = BloomConfig()
            config.num_labels = 1
            model = BloomForSequenceClassification(config)
        if checkpoint:
            model.gradient_checkpointing_enable()

        super().__init__(model, lora_rank, lora_train_bias)
