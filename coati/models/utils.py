from typing import Optional, Union, Dict
import functools

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_base: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        return approx_kl
    approx_kl = approx_kl.mean(dim=1)
    return approx_kl


def compute_reward(r: Union[torch.Tensor, float],
                   kl_coef: float,
                   log_probs: torch.Tensor,
                   log_probs_base: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if kl_coef <= 0.0:
        return r
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    reward = r - kl_coef * kl
    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def normalize(tensor: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    mean = tensor.mean(dim)
    mean_centered = tensor - mean
    var = (mean_centered**2).mean(dim)
    norm = mean_centered * var.clamp(min=eps).rsqrt()
    return norm


def convert_to_lora(model: nn.Module,
                    input_size: int,
                    output_size: int,
                    lora_rank: int = 16,
                    lora_alpha: int = 1,
                    lora_dropout: float = 0.,
                    fan_in_fan_out: bool = False,
                    merge_weights: bool = True):
    if lora_rank > min(input_size, output_size):
        raise ValueError(f"LoRA rank {lora_rank} must be less or equal than {min(input_size, output_size)}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._modules[name] = lora.Linear(input_size,
                                                output_size,
                                                r=lora_rank,
                                                lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout,
                                                fan_in_fan_out=fan_in_fan_out,
                                                merge_weights=merge_weights)

def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])

def rgetattr(obj, attr: str, *args):
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)

def hf_get_causal_hidden_layers(model: nn.Module):
    """Returns the hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = (
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "transformer.layers",
    )
    return findattr(model, hidden_layers_attrs)

def add_tokens(model, tokenizer, tokens: Dict[str, str]):
    model = model.model
    
    tokenizer.add_tokens(list(tokens.keys()))
    model.resize_token_embeddings(len(tokenizer))

    for token, token_init in tokens.items():
        
        index = tokenizer.encode(token, add_special_tokens=False)
        index_init = tokenizer.encode(token_init, add_special_tokens=False)

        assert len(index) == 1
        assert len(index_init) == 1, tokenizer.decode(index_init)
        
        embeddings = model.get_input_embeddings()
        embeddings.weight.data[index] = embeddings.weight.data[index_init]

        token_init = token_init.replace('\n', '\\n')
        logger.info(f'Adding token "{token}", its embedding is initialized to "{token_init}".', ranks=[0])
