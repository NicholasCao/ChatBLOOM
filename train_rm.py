import argparse
import random
from random import randint
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import loralib as lora
import torch
from coati.dataset import RMDataset, DataCollatorForRMDataset
from coati.dataset.utils import jload
from coati.models import LogExpLoss, LogSigLoss, add_tokens
from coati.models.base import RewardModel
from coati.models.bloom import BLOOMRM
from coati.models.deberta import DebertaRM
from coati.models.gpt import GPTRM
from coati.models.llama import LlamaRM
from coati.models.opt import OPTRM
from coati.models.roberta import RoBERTaRM
from coati.trainer import RewardModelTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from datasets import load_dataset
from torch.optim import Adam
import torch.distributed as dist
from transformers import AutoTokenizer, BloomTokenizerFast, DebertaV2Tokenizer, LlamaTokenizer, RobertaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from torch.utils.data.distributed import DistributedSampler

from colossalai.nn.optimizer import HybridAdam
from colossalai.logging import get_dist_logger

from torch.utils.data import DataLoader

def train(args):
    logger = get_dist_logger()

    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BLOOMRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'opt':
            model = OPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'gpt2':
            model = GPTRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'deberta':
            model = DebertaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'llama':
            model = LlamaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        elif args.model == 'roberta':
            model = RoBERTaRM(pretrained=args.pretrain, lora_rank=args.lora_rank).to(torch.cuda.current_device())
        else:
            raise ValueError(f'Unsupported model "{args.model}"')

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.pretrain)
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
    elif args.model == 'deberta':
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.pretrain)
    elif args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(args.pretrain)
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrain)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    max_len = args.max_len

    if args.model == 'llama':
        tokenizer = prepare_llama_tokenizer_and_embedding(tokenizer, model)
    else:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.truncation_side = 'left'
    
    add_tokens(model, tokenizer, {
        '<Human>': ' Human',
        '<Assistant>': ' Assistant',
        '<eoh>': '\n',
        '<eoa>': '\n'
    })

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=args.lr)
    else:
        optim = Adam(model.parameters(), lr=args.lr)

    # configure loss function
    if args.loss_fn == 'log_sig':
        loss_fn = LogSigLoss()
    elif args.loss_fn == 'log_exp':
        loss_fn = LogExpLoss()
    else:
        raise ValueError(f'Unsupported loss function "{args.loss_fn}"')

    # prepare for data and dataset
    train_data = jload(args.data_path)

    random.shuffle(train_data)

    evel_data = train_data[:1000] # split dev set
    train_data = train_data[1000:]

    train_dataset = RMDataset(train_data, tokenizer, max_len)
    eval_dataset = RMDataset(evel_data, tokenizer, max_len)

    data_collator = DataCollatorForRMDataset(tokenizer=tokenizer)
    
    train_sampler = None
    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42, drop_last=True)

    train_dataloader = DataLoader(train_dataset,
                                        shuffle=(train_sampler is None),
                                        sampler=train_sampler,
                                        batch_size=args.batch_size,
                                        collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, collate_fn=data_collator)

    trainer = RewardModelTrainer(model=model,
                                 strategy=strategy,
                                 optim=optim,
                                 loss_fn=loss_fn,
                                 train_dataloader=train_dataloader,
                                 eval_dataloader=eval_dataloader,
                                 batch_size=args.batch_size,
                                 max_epochs=args.max_epochs,
                                 accumulation_steps=args.accumulation_steps)

    trainer.fit(logger)
    # save model checkpoint after fitting on only rank0
    trainer.save_model(path=args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'deberta', 'llama', 'roberta'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='data/generated_rm_data.json')
    parser.add_argument('--save_path', type=str, default='outputs/rm_model')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--loss_fn', type=str, default='log_sig', choices=['log_sig', 'log_exp'])
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    train(args)
