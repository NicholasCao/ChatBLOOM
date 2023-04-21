import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, PromptDataset
from coati.models.bloom import BLOOMRM, BLOOMActor, BLOOMCritic
from coati.models.gpt import GPTRM, GPTActor, GPTCritic
from coati.models.llama import LlamaActor, LlamaCritic, LlamaRM
from coati.models.opt import OPTRM, OPTActor, OPTCritic
from coati.models.roberta import RoBERTaRM, RoBERTaActor, RoBERTaCritic
from coati.trainer import PPOTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, LlamaTokenizer, RobertaTokenizer

from colossalai.nn.optimizer import HybridAdam


def main(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2**5)
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    if args.rm_path is not None:
        state_dict = torch.load(args.rm_path, map_location='cpu')

    # configure model
    if args.model == 'gpt2':
        initial_model = GPTActor(pretrained=args.pretrain)
    elif args.model == 'bloom':
        initial_model = BLOOMActor(pretrained=args.pretrain)
    elif args.model == 'opt':
        initial_model = OPTActor(pretrained=args.pretrain)
    elif args.model == 'llama':
        initial_model = LlamaActor(pretrained=args.pretrain)
    elif args.model == 'roberta':
        initial_model = RoBERTaActor(pretrained=args.pretrain)
    else:
        raise ValueError(f'Unsupported actor model "{args.model}"')

    if args.rm_model == None:
        rm_model_name = args.model
    else:
        rm_model_name = args.rm_model

    if rm_model_name == 'gpt2':
        reward_model = GPTRM(pretrained=args.rm_pretrain)
    elif rm_model_name == 'bloom':
        reward_model = BLOOMRM(pretrained=args.rm_pretrain)
    elif rm_model_name == 'opt':
        reward_model = OPTRM(pretrained=args.rm_pretrain)
    elif rm_model_name == 'llama':
        reward_model = LlamaRM(pretrained=args.rm_pretrain)
    elif rm_model_name == 'roberta':
        reward_model = RoBERTaRM(pretrained=args.rm_pretrain)
    else:
        raise ValueError(f'Unsupported reward model "{rm_model_name}"')

    if args.rm_path is not None:
        reward_model.load_state_dict(state_dict)

    if args.strategy != 'colossalai_gemini':
        initial_model.to(torch.float16).to(torch.cuda.current_device())
        reward_model.to(torch.float16).to(torch.cuda.current_device())

    with strategy.model_init_context():
        if args.model == 'gpt2':
            actor = GPTActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'bloom':
            actor = BLOOMActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'opt':
            actor = OPTActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'llama':
            actor = LlamaActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        elif args.model == 'roberta':
            actor = RoBERTaActor(pretrained=args.pretrain, lora_rank=args.lora_rank)
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')

        if rm_model_name == 'gpt2':
            critic = GPTCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'bloom':
            critic = BLOOMCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'opt':
            critic = OPTCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'llama':
            critic = LlamaCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        elif rm_model_name == 'roberta':
            critic = RoBERTaCritic(pretrained=args.rm_pretrain, lora_rank=args.lora_rank, use_action_mask=True)
        else:
            raise ValueError(f'Unsupported reward model "{rm_model_name}"')

        if args.rm_path is not None:
            critic.load_state_dict(state_dict)
            del state_dict

    if args.strategy != 'colossalai_gemini':
        critic.to(torch.float16).to(torch.cuda.current_device())
        actor.to(torch.float16).to(torch.cuda.current_device())

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        actor_optim = HybridAdam(actor.parameters(), lr=args.lr)
        critic_optim = HybridAdam(critic.parameters(), lr=args.lr)
    else:
        actor_optim = Adam(actor.parameters(), lr=args.lr)
        critic_optim = Adam(critic.parameters(), lr=args.lr)

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.pretrain)
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    elif args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.pretrain)
        tokenizer.eos_token = '<\s>'
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrain)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    if args.model == 'llama':
        tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, actor)
    else:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'left'

    prompt_dataset = PromptDataset(tokenizer=tokenizer, data_path=args.prompt_path, max_length=args.instruction_max_length, max_datasets_size=args.max_datasets_size)
    if dist.is_initialized() and dist.get_world_size() > 1:
        prompt_sampler = DistributedSampler(prompt_dataset, shuffle=True, seed=42, drop_last=True)
    prompt_dataloader = DataLoader(prompt_dataset,
                                   shuffle=(prompt_sampler is None),
                                   sampler=prompt_sampler,
                                   batch_size=args.train_batch_size)

    if args.ptx_coef > 0:
        raise NotImplementedError()
        # TODO
        # pretrain_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=args.pretrain_dataset, max_datasets_size=16384)
        # if dist.is_initialized() and dist.get_world_size() > 1:
        #     pretrain_sampler = DistributedSampler(pretrain_dataset, shuffle=True, seed=42, drop_last=True)
        # pretrain_dataloader = DataLoader(pretrain_dataset,
        #                                 shuffle=(pretrain_sampler is None),
        #                                 sampler=pretrain_sampler,
        #                                 batch_size=args.ptx_batch_size,
        #                                 collate_fn=data_collator)
        # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        pretrain_dataloader = None

    # def tokenize_fn(texts):
    #     # MUST padding to max length to ensure inputs of all ranks have the same length
    #     # Different length may lead to hang when using gemini, as different generation steps
    #     batch = tokenizer(texts, return_tensors='pt', max_length=96, padding='max_length', truncation=True)
    #     return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    (actor, actor_optim), (critic, critic_optim) = strategy.prepare((actor, actor_optim), (critic, critic_optim))

    # configure trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        actor_optim,
        critic_optim,
        kl_coef=args.kl_coef,
        ptx_coef=args.ptx_coef,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        experience_batch_size=args.experience_batch_size,
        tokenizer=None,
        max_length=args.max_length,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    trainer.fit(prompt_dataloader=prompt_dataloader,
                pretrain_dataloader=pretrain_dataloader,
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)

    # save model checkpoint after fitting
    trainer.save_model(args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(actor_optim,
                                'actor_optim_checkpoint_prompts_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default=None, help='path to the prompt dataset')
    parser.add_argument('--pretrain_dataset', type=str, default=None, help='path to the pretrained dataset')
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive',
                        help='strategy to use')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'llama', 'roberta'])
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--rm_model', default=None, choices=['gpt2', 'bloom', 'opt', 'llama', 'roberta'])
    parser.add_argument('--rm_path', type=str, default=None)
    parser.add_argument('--rm_pretrain', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='outputs/ppo')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--ptx_batch_size', type=int, default=1)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--ptx_coef', type=float, default=0.0)
    parser.add_argument('--instruction_max_length', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=256)
    
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--accimulation_steps', type=int, default=1)
    parser.add_argument('--max_datasets_size', type=int, default=None)
    args = parser.parse_args()
    main(args)
