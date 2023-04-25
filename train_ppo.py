import argparse
import os
import copy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, PromptDataset, DataCollatorForPromptDataset
from coati.models.bloom import BLOOMRM, BLOOMActor, BLOOMCritic, BLOOMPPO
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


def create_reference_model(model):
    ref_model = copy.deepcopy(model)
    
    for n, param in ref_model.named_parameters():
        param.requires_grad = False

    return ref_model.eval()

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

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BLOOMPPO(pretrained=args.pretrain)
        else:
            raise ValueError(f'Unsupported actor model "{args.model}"')
    
    ref_model = create_reference_model(model)

    if args.rm_model == None:
        rm_model_name = args.model
    else:
        rm_model_name = args.rm_model

    with strategy.model_init_context():
        if rm_model_name == 'bloom':
            reward_model = BLOOMRM(pretrained=args.rm_pretrain)
        else:
            raise ValueError(f'Unsupported reward model "{rm_model_name}"')

    if args.strategy != 'colossalai_gemini':
        model.to(torch.float16).to(torch.cuda.current_device())
        ref_model.to(torch.float16).to(torch.cuda.current_device())
        reward_model.to(torch.float16).to(torch.cuda.current_device())

    reward_model.eval()
    for n, param in reward_model.named_parameters():
        param.requires_grad = False

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=args.lr)
    else:
        optim = Adam(model.parameters(), lr=args.lr)

    # configure tokenizer
    if args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    if rm_model_name == 'bloom':
        rm_tokenizer = BloomTokenizerFast.from_pretrained(args.rm_pretrain)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    tokenizer.truncation_side = 'left'
    rm_tokenizer.truncation_side = 'left'
    
    # TODO
    prompt_dataset = PromptDataset(tokenizer=tokenizer, data_path=args.prompt_path, max_length=args.instruction_max_length, max_datasets_size=args.max_datasets_size)
    if dist.is_initialized() and dist.get_world_size() > 1:
        prompt_sampler = DistributedSampler(prompt_dataset, shuffle=True, seed=42, drop_last=True)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer)
    prompt_dataloader = DataLoader(prompt_dataset,
                                   shuffle=(prompt_sampler is None),
                                   sampler=prompt_sampler,
                                   batch_size=args.batch_size,
                                   collate_fn=data_collator)
    
    

    if args.ptx_coef > 0:
        raise NotImplementedError()
        # TODO
    else:
        pretrain_dataloader = None

    (model, optim) = strategy.prepare((model, optim))
        
    trainer = PPOTrainer(
        model,
        ref_model,
        reward_model,
        strategy,
        optim,
        lr_scheduler=None,
        tokenizer=tokenizer,
        dataloader=prompt_dataloader,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        accumulation_steps=args.accumulation_steps,
        kl_coef=args.kl_coef,
        vf_coef=args.vf_coef
    )

    trainer.fit(rm_tokenizer, path=args.save_path, max_new_tokens=args.max_new_tokens, reward_baseline=args.reward_baseline, save_interval=50)

    # save model checkpoint after fitting
    trainer.save_model(args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(optim,
                                'optim_checkpoint_prompts_%d.pt' % (torch.cuda.current_device()),
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
    parser.add_argument('--rm_pretrain', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='outputs/ppo')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    # parser.add_argument('--generate_time_per_turn', type=int, default=1)
    parser.add_argument('--mini_batch_size', type=int, default=2)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--vf_coef', type=float, default=0.05)
    parser.add_argument('--ptx_coef', type=float, default=0.0)
    parser.add_argument('--instruction_max_length', type=int, default=384)
    parser.add_argument('--max_new_tokens', type=int, default=384)
    
    parser.add_argument('--lr', type=float, default=5e-6)
    
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--max_datasets_size', type=int, default=None)
    parser.add_argument('--reward_baseline', type=int, default=0)
    
    args = parser.parse_args()
    main(args)
