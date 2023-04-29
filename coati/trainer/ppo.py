import math
import time
from abc import ABC
from typing import *
import os
import copy

import loralib as lora
import torch
import torch.distributed as dist
import wandb
from coati.models.loss import GPTLMLoss
import numpy as np
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.trainer import get_scheduler
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

from colossalai.logging import get_dist_logger

from .strategies import Strategy
from .utils import is_rank_0
from .utils import (
    set_seed,
    PPODecorators,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    flatten_dict,
    stack_dicts,
    entropy_from_logits,
    clip_by_value,
    stats_to_np
)


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class PPOTrainer(ABC):
    def __init__(
        self,
        model,
        ref_model,
        reward_model,
        strategy: Strategy,
        optim: Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: PreTrainedTokenizer = None,
        dataloader: DataLoader = None,
        batch_size: int = 4,
        mini_batch_size: int = 2,
        ppo_epochs: int = 4,
        
        accumulation_steps: int = 1,
        kl_coef: float = 0.05,
        kl_target: float = 6,
        kl_horizon: int = 10000,
        
        gamma: float = 1,
        lam: float = 0.95,
        cliprange_value: float = 0.2,
        cliprange: float = 0.2,
        vf_coef : float = 0.1,
    ):
        super().__init__()

        set_seed(42)
        
        self.model = model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.strategy = strategy
        if "DDP" in str(self.strategy):
            self.model = self.model.module

        self.optimizer = optim
        self.lr_scheduler = lr_scheduler

        self.accumulation_steps = accumulation_steps
        self.kl_ctl = AdaptiveKLController(kl_coef, target=kl_target, horizon=kl_horizon)
        self.gamma = gamma
        self.lam = lam
        self.cliprange_value = cliprange_value
        self.cliprange = cliprange
        self.vf_coef = vf_coef
        
        self.current_step = 0
        self.train_step = 0
    
    @PPODecorators.empty_cuda_cache()
    def fit(self, rm_tokenizer, path = None, max_length: int = 768, reward_baseline: int = 0, save_interval: int = 50):
        if is_rank_0():
            wandb.init(project="Coati", name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' PPO')

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_length": max_length,
            "early_stopping": True,
            "temperature": 1.0
        }
        for epoch, batch in enumerate(tqdm(self.dataloader, disable=not is_rank_0())):
            # self.ref_model.to('cpu')
            query_tensors = batch["input_ids"].to(torch.cuda.current_device())

            response_tensors = self.generate(
                query_tensors, return_prompt=False, **generation_kwargs
            )

            batch["query"] = self.tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
            batch["response"] = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            # Compute reward
            texts = [q + r + rm_tokenizer.eos_token for q, r in zip(batch["query"], batch["response"])]
            
            reward_inputs = rm_tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(
                torch.cuda.current_device()
            )
            
            rewards = torch.tensor([]).to(torch.cuda.current_device())
            
            for i in range(int(self.batch_size / self.mini_batch_size)):
                input_kwargs = {key: value[i * self.mini_batch_size : (i + 1) * self.mini_batch_size] for key, value in reward_inputs.items()}
                cur_rewards = self.reward_model(**input_kwargs).float() - reward_baseline
                rewards = torch.cat([rewards, cur_rewards])

            # Run PPO step
            stats = self.step(query_tensors, response_tensors, rewards)
            logs = self.get_log_stats(stats, batch, rewards)
            if is_rank_0():
                wandb.log(logs)
            
            # Save model every 100 epochs
            if (epoch + 1) % save_interval == 0 and path is not None:
                self.save_model(path, only_rank0=True, tokenizer=self.tokenizer)

    def save_model(self,
                   path: str,
                   only_rank0: bool = True,
                   tokenizer: Optional[PreTrainedTokenizer] = None) -> None:

        if is_rank_0() and not os.path.exists(path):
            os.makedirs(path)
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`batch_size`, `seq_len`) containing query tokens.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """

        if isinstance(query_tensor, List) or (isinstance(query_tensor, torch.Tensor) and query_tensor.dim() == 2):
            return self._generate_batched(
                query_tensor,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )

        else:
            response = self.model.generate(
                input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
            )

            if not return_prompt:
                return response[:, query_tensor.shape[0] :]
            return response

    def _generate_batched(
        self,
        query_tensors: List[torch.Tensor],
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        **generation_kwargs,
    ):
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(torch.cuda.current_device())

            generations = self.model.generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                output = generation[(1 - mask).sum() :]  # remove padding

                if not return_prompt:
                    output = output[(mask).sum() :]  # remove prompt
                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    @PPODecorators.empty_cuda_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.batch_size

        model_inputs = self.prepare_model_inputs(queries, responses)

        model_inputs_names = list(model_inputs.keys())

        with torch.no_grad():
            all_logprobs, _, values, masks = self.batched_forward_pass(self.model, queries, responses, model_inputs)
            # self.ref_model.to(torch.cuda.current_device())
            ref_logprobs, _, _, _ = self.batched_forward_pass(self.ref_model, queries, responses, model_inputs)
            # self.ref_model.to('cpu')

        rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)

        # upcast to float32 to avoid dataset issues
        mini_batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "rewards": rewards,
            "masks": masks,
        }

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                if key in ["queries", "responses"]:
                    return_dict[key] = [d[key] for d in data]
                else:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(torch.cuda.current_device())
            return return_dict

        mini_batch_dict.update(model_inputs)
        mini_batch_data = Dataset.from_dict(mini_batch_dict)
        mini_batch_data.set_format("torch")
        mini_batch_dataloader = torch.utils.data.DataLoader(
            mini_batch_data,
            batch_size=self.mini_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        all_stats = []
        early_stop = False
        for _ in range(self.ppo_epochs):
            if early_stop:
                break
            for batch in mini_batch_dataloader:
                model_inputs = {k: batch[k] for k in model_inputs_names}
                logprobs, logits, vpreds, _ = self.batched_forward_pass(
                    self.model, batch["queries"], batch["responses"], model_inputs, return_logits=True
                )

                train_stats = self.train_minibatch(
                    batch["logprobs"],
                    batch["values"],
                    batch["rewards"],
                    logprobs,
                    logits,
                    vpreds,
                    batch["masks"],
                )

                all_stats.append(train_stats)

        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], -1)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )

        stats = stats_to_np(stats)
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(stats["objective/kl"], self.batch_size * torch.distributed.get_world_size())

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats
    
    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        input_data = self.data_collator(
            [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
        ).to(queries.device)

        input_data.pop("labels", None)  # we don't want to compute LM losses

        return input_data
    
    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            logits = logits.to(torch.float32)
            values = values.to(torch.float32)

            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(fbs):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0]
                end = start + len(response_batch[j])

                if len(logprobs[j, start:end]) < 2:
                    raise ValueError("Responses are too short. Make sure they are at least 4 tokens long.")

                masks[j, :start] = 0
                masks[j, end:] = 0

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards = [], []
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # compute KL penalty (from difference in logprobs)
            kl = logprob - ref_logprob
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards)

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`, `vocab_size`)
            v_pred (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds, values - self.cliprange_value, values + self.cliprange_value
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).double(), mask)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).double(), mask)

        loss = pg_loss + self.vf_coef * vf_loss

        entropy = masked_mean(entropy_from_logits(logits), mask)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)
        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.vf_coef * vf_loss, flatten_dict(stats)
    
    @PPODecorators.empty_cuda_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [batch_size, response_length]
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape [batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        loss_p, loss_v, train_stats = self.loss(old_logprobs, values, rewards, logits, vpreds, logprobs, mask)
        loss = loss_p + loss_v

        loss = loss / self.accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
        )

        if (self.train_step + 1) % self.accumulation_steps == 0:
            self.strategy.optimizer_step(self.optimizer)
            self.optimizer.zero_grad()

        self.train_step += 1
        return train_stats

    def record_step_stats(self, kl_coef: float, **data):
        """
        Record training step statistics.

        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
        mask = data.pop("masks")

        kl_list = ((data["logprobs"] - data["ref_logprobs"]) * mask).sum(axis=-1)
        mean_kl = kl_list.mean()
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

        mean_non_score_reward = masked_mean(
            data["non_score_reward"], mask
        )  # non_score_reward is size `batch_size`, `response_length`
        mean_scores = data["scores"].mean()  # scores is size `batch_size`
        std_scores = data["scores"].std()

        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
            "ppo/mean_scores": mean_scores,
            "ppo/std_scores": std_scores,
        }

        # Log text properties
        query_lens = torch.tensor([len(query) for query in data["queries"]], dtype=torch.float)
        response_lens = torch.tensor([len(response) for response in data["responses"]], dtype=torch.float)

        stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item()
        stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item()
        stats["tokens/queries_dist"] = query_lens.cpu().numpy()
        stats["tokens/responses_len_mean"] = torch.mean(response_lens).cpu().numpy().item()
        stats["tokens/responses_len_std"] = torch.std(response_lens).cpu().numpy().item()
        stats["tokens/responses_dist"] = response_lens.cpu().numpy()

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        return stats

    def get_log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """
        # Log only if we are in the main process
        logs = {}

        # Log stats
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(self.current_device)

        table_rows = [list(r) for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())]
        logs.update({"game_log": wandb.Table(columns=["query", "response", "reward"], rows=table_rows)})

        logs.update(stats)

        # manually cast in fp32 for bf16 torch tensors
        for k, v in logs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                logs[k] = v.float()

        logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
        logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
        logs["env/reward_dist"] = rewards.cpu().numpy()

        logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
        logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
        logs["env/reward_dist"] = rewards.cpu().numpy()
        return logs    
