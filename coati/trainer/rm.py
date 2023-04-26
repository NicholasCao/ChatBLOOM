from abc import ABC
import os
import math
from datetime import datetime
import time
from typing import Optional

import wandb
import pandas as pd
import torch
import torch.distributed as dist
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import get_scheduler

from .strategies import Strategy
from .utils import is_rank_0


class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        loss_fn (callable): the loss function to use for training
        train_dataset (Dataset): the dataset to use for training
        eval_dataset (Dataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        loss_fn,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        batch_size: int = 1,
        max_epochs: int = 1,
        accumulation_steps: int = 1,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module
        self.loss_fn = loss_fn
        self.optimizer = strategy.setup_optimizer(optim, self.model)
        
        self.accumulation_steps = accumulation_steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.accumulation_steps
        max_steps = math.ceil(self.epochs * num_update_steps_per_epoch)
        self.scheduler = get_scheduler("cosine",
                                self.optimizer,
                                num_warmup_steps=math.ceil(max_steps * 0.03),
                                num_training_steps=max_steps)

    def eval_acc(self, dataloader):
        dist = 0
        on = 0
        cnt = 0
        self.model.eval()
        rewards = torch.tensor([]).to(torch.cuda.current_device())
        with torch.no_grad():
            for batch in dataloader:
                chosen_ids = batch["chosen_input_ids"].to(torch.cuda.current_device())
                c_mask = batch["chosen_attention_mask"].to(torch.cuda.current_device())
                reject_ids = batch["reject_input_ids"].to(torch.cuda.current_device())
                r_mask = batch["reject_attention_mask"].to(torch.cuda.current_device())

                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                
                rewards = torch.cat([rewards, chosen_reward, reject_reward])
                for i in range(len(chosen_reward)):
                    cnt += 1
                    if chosen_reward[i] > reject_reward[i]:
                        on += 1
                dist += (chosen_reward - reject_reward).mean().item()
            dist_mean = dist / len(dataloader)
            acc = on / cnt
            reward_mean = rewards.view(-1).mean()
            reward_std = rewards.view(-1).std()
            
        self.model.train()
        return {
            "dist": dist_mean,
            "acc": acc,
            "reward_mean": reward_mean,
            "reward_std": reward_std
        }

    def fit(self, logger):
        if is_rank_0():
            wandb.init(project="Coati", name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' RM')
            wandb.watch(self.model)

        step_bar = tqdm(range(len(self.train_dataloader) // self.accumulation_steps * self.epochs),
                        desc=f'steps',
                        disable=not is_rank_0())
        for epoch in range(self.epochs):
            # train
            self.model.train()
            total_loss = 0
            
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            for batch_id, batch in enumerate(self.train_dataloader):
                chosen_ids = batch["chosen_input_ids"].to(torch.cuda.current_device())
                c_mask = batch["chosen_attention_mask"].to(torch.cuda.current_device())
                reject_ids = batch["reject_input_ids"].to(torch.cuda.current_device())
                r_mask = batch["reject_attention_mask"].to(torch.cuda.current_device())
                
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
    
                loss = self.loss_fn(chosen_reward, reject_reward)
                loss = loss / self.accumulation_steps
                
                self.strategy.backward(loss, self.model, self.optimizer)
                total_loss += loss.item()

                # gradient accumulation
                if (batch_id + 1) % self.accumulation_steps == 0:
                    self.strategy.optimizer_step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    if is_rank_0():
                        wandb.log({
                            "loss": total_loss,
                            "lr": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "batch_id": batch_id
                        })
                    total_loss = 0
                    step_bar.update()
                    
                    if (batch_id // self.accumulation_steps + 1) % 100 == 0:
                        results = self.eval_acc(self.eval_dataloader)
                        if is_rank_0():
                            wandb.log({
                                "dist": results['dist'],
                                "acc": results['acc'],
                                "r_mean": results['reward_mean'],
                                "r_std": results['reward_std'],
                                "batch_id": batch_id
                            })
                        logger.info(f"Eval: dist={results['dist']}, acc={results['acc']}, r_mean={results['reward_mean']}, r_std={results['reward_std']}", ranks=[0])
        step_bar.close()

    def save_model(self,
                   path: str,
                   only_rank0: bool = False,
                   tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)
