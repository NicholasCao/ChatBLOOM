from abc import ABC
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
        valid_dataset (Dataset): the dataset to use for validation
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
        valid_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        batch_size: int = 1,
        max_epochs: int = 1,
        accimulation_steps: int = 1,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.eval_dataloader = eval_dataloader

        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module
        self.loss_fn = loss_fn
        self.optimizer = strategy.setup_optimizer(optim, self.model)
        
        self.accimulation_steps = accimulation_steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.accimulation_steps
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
        with torch.no_grad():
            for batch in dataloader:
                chosen_ids = batch["chosen_input_ids"].to(torch.cuda.current_device())
                c_mask = batch["chosen_attention_mask"].to(torch.cuda.current_device())
                reject_ids = batch["reject_input_ids"].to(torch.cuda.current_device())
                r_mask = batch["reject_attention_mask"].to(torch.cuda.current_device())

                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                for i in range(len(chosen_reward)):
                    cnt += 1
                    if chosen_reward[i] > reject_reward[i]:
                        on += 1
                dist += (chosen_reward - reject_reward).mean().item()
            dist_mean = dist / len(dataloader)
            acc = on / cnt
        self.model.train()
        return dist_mean, acc

    def fit(self, logger):
        if is_rank_0():
            wandb.init(project="Coati", name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            wandb.watch(self.model)

        step_bar = tqdm(range(len(self.train_dataloader) // self.accimulation_steps * self.epochs),
                        desc=f'steps',
                        disable=not is_rank_0())
        for epoch in range(self.epochs):
            # train
            self.model.train()
            acc = 0
            dist = 0
            total_loss = 0

            for batch_id, batch in enumerate(self.train_dataloader):
                chosen_ids = batch["chosen_input_ids"].to(torch.cuda.current_device())
                c_mask = batch["chosen_attention_mask"].to(torch.cuda.current_device())
                reject_ids = batch["reject_input_ids"].to(torch.cuda.current_device())
                r_mask = batch["reject_attention_mask"].to(torch.cuda.current_device())
                # chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                # c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                # reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                # r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
    
                loss = self.loss_fn(chosen_reward, reject_reward)
                loss = loss / self.accimulation_steps
                
                self.strategy.backward(loss, self.model, self.optimizer)
                total_loss += loss.item()

                # gradient accumulation
                if (batch_id + 1) % self.accimulation_steps == 0:
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
                    
                    if (batch_id // self.accimulation_steps + 1) % 100 == 0:
                        dist, acc = self.eval_acc(self.valid_dataloader)
                        if is_rank_0():
                            wandb.log({
                                "dist": dist,
                                "acc": acc,
                                "batch_id": batch_id
                            })
                        logger.info(f'Eval dev: dist={dist}, acc={acc}', ranks=[0])

            # eval
            dist, acc = self.eval_acc(self.eval_dataloader)
            logger.info(f'Eval test: dist={dist}, acc={acc}', ranks=[0])
            step_bar.set_postfix({'dist': dist, 'acc': acc})
        step_bar.close()

    def save_model(self,
                   path: str,
                   only_rank0: bool = False,
                   tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)
