import json
import math
import os
import sys
import io

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=768,
        epochs=10000,
        total_steps=2500,
        batch_size=4,
        checkpoint_interval=1000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="outputs/bloom-1b7-ppo",
        save_optimizer=False
    ),
    model=ModelConfig(model_path="outputs/bloom-1b7-sft", num_layers_unfrozen=4),
    tokenizer=TokenizerConfig(tokenizer_path="outputs/bloom-1b7-sft", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=5e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=5e-6)),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=4,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=320,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
    ),
)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

rm_model_path = 'outputs/bloom-1b7-rm'

def create_reward_fn():  # noqa:  C901
    reward_tokenizer = AutoTokenizer.from_pretrained(rm_model_path)
    # reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    reward_tokenizer.padding_side = "right"

    if os.environ.get("RANK", "0") == "0":
        class RewardModel(nn.Module):
            def __init__(self, checkpoint_path):
                super().__init__()
                self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

            def forward(self, input_ids: torch.LongTensor, attention_mask = None) -> torch.Tensor:
                outputs = self.model(input_ids, attention_mask=attention_mask)
                value = outputs['logits'].squeeze(-1)
                return value
                    
        reward_model = RewardModel(rm_model_path)

        reward_model.eval()
        reward_model.requires_grad_(False)
        reward_device = torch.cuda.device_count() - 1
        reward_model = reward_model.half().to(reward_device)
        reward_batch_size = 64
        delta_reward = True

        def get_reward(samples):
            input = reward_tokenizer(
                samples,
                padding=True,
                truncation=True,
                max_length=768,
                return_tensors="pt",
            ).to(reward_device)

            mbs = reward_batch_size
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = input.input_ids[batch_ixs]
                attention_mask = input.attention_mask[batch_ixs]
                rewards = reward_model(input_ids, attention_mask)
                out.extend(rewards)
            return torch.hstack(out)

        def reward_fn(samples, prompts, original_output, **kwargs):
            # samples = [s + reward_tokenizer.eos_token for s in samples]
            # Fix: eos_token is appended in trainer
            samples = [s for s in samples]
            rewards = get_reward(samples)

            if not delta_reward:
                return rewards

            original_samples = [p + o + reward_tokenizer.eos_token for p, o in zip(prompts, original_output)]
            # original_samples = [p + o for p, o in zip(prompts, original_output)]
            original_rewards = get_reward(original_samples)
            return rewards - original_rewards

    else:
        reward_fn = True

    return reward_fn

def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    dataset = jload('data/prompt_data.json')
    prompts = [{"prompt": x["query"], "original_output": x["response"]} for x in dataset[500:]]
    eval_prompts = [{"prompt": x["query"], "original_output": x["response"]} for x in dataset[:500]]
    reward_fn = create_reward_fn()

    trainer = trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["<Human>", "<Assistant>"],
    )
    trainer.save_pretrained("outputs/bloom-1b7-ppo/hf_model")


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
