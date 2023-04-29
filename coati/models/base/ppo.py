from typing import Optional

import torch
import torch.nn as nn

from ..lora import LoRAModule
from ..utils import hf_get_causal_hidden_layers

class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for CausalLM that returns a scalar for each output token.
    """

    def __init__(self, config, dropout_prob=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output

class PPOModel(LoRAModule):
    """
    PPO model base class.

    Args:
        model (nn.Module): Critic model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_rank: int = 0,
        lora_train_bias: str = 'none',
        freeze_layer_ratio: float = 0.67,
    ) -> None:

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

        self.v_head = ValueHead(self.model.config)
        
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ):

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        last_hidden_states = outputs['hidden_states'][-1]
        lm_logits = outputs.logits
        loss = outputs.loss
        
        if last_hidden_states.device != self.v_head.summary.weight.device:
            last_hidden_states = last_hidden_states.to(self.v_head.summary.weight.device)
        value = self.v_head(last_hidden_states).squeeze(-1)
        
        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
        if value.dtype != torch.float32:
            value = value.float()

        return (lm_logits, loss, value)

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """

        return self.model.generate(*args, **kwargs)
