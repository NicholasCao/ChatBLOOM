from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
import os

import wandb
import torch
from torch import Tensor
from torch.utils.data import DistributedSampler
import torch.nn as nn
from tqdm import tqdm
from coati.experience_maker import Experience, ExperienceMaker, NaiveExperienceMaker
from coati.models.base import Actor, Critic
from coati.models.generation_utils import update_model_kwargs_fn
from coati.models.loss import PolicyLoss, ValueLoss
from coati.replay_buffer import NaiveReplayBuffer, ReplayBuffer
from torch.optim import Optimizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .callbacks import Callback
from .strategies import Strategy
from .utils import is_rank_0


class PPOTrainer(ABC):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (Critic): the critic model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        vf_coef (float, defaults to 1.0): the coefficient of value loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(self,
                 strategy: Strategy,
                 actor: Actor,
                 critic: Critic,
                 reward_model: nn.Module,
                 initial_model: Actor,
                 actor_optim: Optimizer,
                 critic_optim: Optimizer,
                 kl_coef: float = 0.1,
                 ptx_coef: float = 0.9,
                 train_batch_size: int = 8,
                 buffer_limit: int = 0,
                 buffer_cpu_offload: bool = True,
                 eps_clip: float = 0.2,
                 vf_coef: float = 1.0,
                 value_clip: float = 0.4,
                 experience_batch_size: int = 8,
                 max_epochs: int = 1,
                 sample_replay_buffer: bool = False,
                 dataloader_pin_memory: bool = True,
                 callbacks: List[Callback] = [],
                 accimulation_steps: int = 1,
                 **generate_kwargs) -> None:
        experience_maker = NaiveExperienceMaker(actor, critic, reward_model, initial_model, kl_coef)
        replay_buffer = NaiveReplayBuffer(train_batch_size, buffer_limit, buffer_cpu_offload)
        generate_kwargs = _set_default_generate_kwargs(strategy, generate_kwargs, actor)

        self.strategy = strategy
        self.experience_maker = experience_maker
        self.replay_buffer = replay_buffer
        self.experience_batch_size = experience_batch_size
        self.max_epochs = max_epochs
        self.generate_kwargs = generate_kwargs
        self.sample_replay_buffer = sample_replay_buffer
        self.dataloader_pin_memory = dataloader_pin_memory
        self.callbacks = callbacks
        self.accimulation_steps = accimulation_steps

        self.actor = actor
        self.critic = critic

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.vf_coef = vf_coef
        self.ptx_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.ptx_coef = ptx_coef
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
    
    def _make_experience(self, inputs: Union[Tensor, Dict[str, Tensor]]) -> Experience:
        if isinstance(inputs, Tensor):
            return self.experience_maker.make_experience(inputs, **self.generate_kwargs)
        elif isinstance(inputs, dict):
            return self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
        else:
            raise ValueError(f'Unsupported input type "{type(inputs)}"')
    
    def _sample_prompts(self, prompts) -> list:
        indices = list(range(len(prompts)))
        sampled_indices = self.strategy.experience_sampler.choice(indices, self.experience_batch_size, replace=False)
        return [prompts[i] for i in sampled_indices]

    def _learn(self):
        # replay buffer may be empty at first, we should rebuild at each training
        if not self.sample_replay_buffer:
            dataloader = self.strategy.setup_dataloader(self.replay_buffer, self.dataloader_pin_memory)
            device = torch.cuda.current_device()
        if self.sample_replay_buffer:
            pbar = tqdm(range(self.max_epochs), desc='Train epoch', disable=not is_rank_0())
            for i in pbar:
                experience = self.replay_buffer.sample()
                metrics = self.training_step(experience, step=(i + 1) % self.accimulation_steps == 0)
                pbar.set_postfix(metrics)
        else:
            for epoch in range(self.max_epochs):
                self._on_learn_epoch_start(epoch)
                if isinstance(dataloader.sampler, DistributedSampler):
                    dataloader.sampler.set_epoch(epoch)
                pbar = tqdm(dataloader, desc=f'Train epoch [{epoch+1}/{self.max_epochs}]', disable=not is_rank_0())
                for i, experience in enumerate(pbar):
                    self._on_learn_batch_start()
                    experience.to_device(device)
                    metrics = self.training_step(experience, step=(i + 1) % self.accimulation_steps == 0)
                    self._on_learn_batch_end(metrics, experience)
                    pbar.set_postfix(metrics)
                self._on_learn_epoch_end(epoch)
    
    def training_step(self, experience: Experience, step: bool = True) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()
        # policy loss
        num_actions = experience.action_mask.size(1)
        action_log_probs = self.actor(experience.sequences, num_actions, attention_mask=experience.attention_mask)
        actor_loss = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask)

        # ptx loss
        if self.ptx_coef != 0:
            batch = next(iter(self.pretrain_dataloader))
            ptx = batch['input_ids'].to(torch.cuda.current_device())
            label = batch['labels'].to(torch.cuda.current_device())[:, 1:]
            attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
            ptx_log_probs = self.actor.get_base_model()(ptx, attention_mask=attention_mask)['logits'][..., :-1, :]
            ptx_loss = self.ptx_loss_fn(ptx_log_probs.view(-1, ptx_log_probs.size(-1)), label.view(-1))
            actor_loss = ptx_loss * self.ptx_coef + actor_loss * (1 - self.ptx_coef)

        self.strategy.backward(actor_loss, self.actor, self.actor_optim)
        
        if step:
            self.strategy.optimizer_step(self.actor_optim)
            self.actor_optim.zero_grad()

        # value loss
        values = self.critic(experience.sequences,
                             action_mask=experience.action_mask,
                             attention_mask=experience.attention_mask)
        critic_loss = self.critic_loss_fn(values,
                                          experience.values,
                                          experience.reward,
                                          action_mask=experience.action_mask)
        critic_loss = critic_loss * self.vf_coef
        self.strategy.backward(critic_loss, self.critic, self.critic_optim)
        if step:
            self.strategy.optimizer_step(self.critic_optim)
            self.critic_optim.zero_grad()
            if is_rank_0():
                wandb.log({
                    "reward": experience.reward.mean().item(),
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "lr": self.scheduler.get_last_lr()[0],
                })

        return {'reward': experience.reward.mean().item()}

    def fit(self,
            prompt_dataloader,
            pretrain_dataloader,
            num_episodes: int = 1,
            max_timesteps: int = -1,
            update_timesteps: int = 10) -> None:
        if is_rank_0():
            wandb.init(project="Coati", name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            wandb.watch(self.model)

        time = 0
        self.pretrain_dataloader = pretrain_dataloader
        self.prompt_dataloader = prompt_dataloader
        self._on_fit_start()
        for episode in range(num_episodes):
            self._on_episode_start(episode)
            if isinstance(self.prompt_dataloader.sampler, DistributedSampler):
                self.prompt_dataloader.sampler.set_epoch(episode)
            for prompts in tqdm(self.prompt_dataloader,
                                desc=f'Episode [{episode+1}/{num_episodes}]',
                                disable=not is_rank_0()):
                time += 1
                # prompts = next(iter(self.prompt_dataloader))
                self._on_make_experience_start()
                self.experience_maker.initial_model.to(torch.cuda.current_device())
                self.experience_maker.reward_model.to(torch.cuda.current_device())
                experience = self._make_experience(prompts)
                self._on_make_experience_end(experience)
                self.replay_buffer.append(experience)
                if time % update_timesteps == 0:
                    self.experience_maker.initial_model.to('cpu')
                    self.experience_maker.reward_model.to('cpu')
                    self._learn()
                    self.replay_buffer.clear()
        self._on_episode_end(episode)
        self._on_fit_end()
    
    def save_model(self, path: str, only_rank0: bool = False, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        if is_rank_0() and not os.path.exists(path):
            os.makedirs(path)
        self.strategy.save_model(model=self.actor, path=path, only_rank0=only_rank0, tokenizer=tokenizer)
    
    # TODO(ver217): maybe simplify these code using context
    def _on_fit_start(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_start()

    def _on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end()

    def _on_episode_start(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(episode)

    def _on_episode_end(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(episode)

    def _on_make_experience_start(self) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_start()

    def _on_make_experience_end(self, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_end(experience)

    def _on_learn_epoch_start(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_start(epoch)

    def _on_learn_epoch_end(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_end(epoch)

    def _on_learn_batch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_start()

    def _on_learn_batch_end(self, metrics: dict, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_end(metrics, experience)


def _set_default_generate_kwargs(strategy: Strategy, generate_kwargs: dict, actor: Actor) -> None:
    origin_model = strategy._unwrap_actor(actor)
    new_kwargs = {**generate_kwargs}
    # use huggingface models method directly
    if 'prepare_inputs_fn' not in generate_kwargs and hasattr(origin_model, 'prepare_inputs_for_generation'):
        new_kwargs['prepare_inputs_fn'] = origin_model.prepare_inputs_for_generation

    if 'update_model_kwargs_fn' not in generate_kwargs:
        new_kwargs['update_model_kwargs_fn'] = update_model_kwargs_fn

    return new_kwargs
