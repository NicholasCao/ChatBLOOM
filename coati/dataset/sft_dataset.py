#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Optional, List
import re

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset

from colossalai.logging import get_dist_logger

from .utils import is_rank_0, jload

logger = get_dist_logger()

IGNORE_INDEX = -100

def pCLUE_preprocess(data_path: str, tokenizer: Callable, max_length: int = 512, max_datasets_size: int = 600000):
    dataset = load_dataset(data_path, split='train')
    
    input_ids = []
    input_lens = []
    labels = []
    for idx, data in enumerate(tqdm(dataset, disable=not is_rank_0(), mininterval=3, desc=f'preprocssing {data_path}')):
        
        input_text = data['input'] + data['target'] + tokenizer.eos_token
        
        inputs = tokenizer(input_text,
                           max_length=max_length,
                           padding=False,
                           truncation=True,
                           return_tensors="pt")
        if len(inputs['input_ids'][0]) < 20 or len(inputs['input_ids'][0]) >= max_length - 1:
            continue
        
        input_len = len(tokenizer.tokenize(data['input']))
        
        input_ids.append(inputs['input_ids'][0])
        input_lens.append(input_len)
        
        if max_datasets_size is not None and idx >= max_datasets_size:
            break
    
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, input_lens):
        label[:source_len] = IGNORE_INDEX
    return input_ids, labels

def BELLE_preprocess(data_path: str, tokenizer: Callable, max_length: int = 512, max_datasets_size: int = 1000000, require_enter: bool = False):
    dataset = load_dataset(data_path, split='train')
    
    input_ids = []
    input_lens = []
    labels = []
    
    enter = '\n' if require_enter else ''
    for idx, data in enumerate(tqdm(dataset, disable=not is_rank_0(), mininterval=3, desc=f'preprocssing {data_path}')):
        
        input_text = data['instruction'] + enter + data['output'] + tokenizer.eos_token
        
        inputs = tokenizer(input_text,
                           max_length=max_length,
                           padding=False,
                           truncation=True,
                           return_tensors="pt")
        if len(inputs['input_ids'][0]) < 20 or len(inputs['input_ids'][0]) >= max_length - 1:
            continue
        
        input_len = len(tokenizer.tokenize(data['instruction']))
        
        input_ids.append(inputs['input_ids'][0])
        input_lens.append(input_len)
        
        if max_datasets_size is not None and idx >= max_datasets_size:
            break

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, input_lens):
        label[:source_len] = IGNORE_INDEX
    return input_ids, labels
        
class ITDataset(Dataset):
    """
    Dataset for instruction tuning

    Args:
        tokenizer: tokenizer for supervised model
        max_length: max length of input
    """

    def __init__(self, tokenizer: Callable, max_length: int = 512) -> None:
        super().__init__()
        self.input_ids = []
        self.labels = []
        
        input_ids, labels = BELLE_preprocess('BelleGroup/generated_chat_0.4M', tokenizer, max_length, 200000)
        self.input_ids += input_ids
        self.labels += labels

        input_ids, labels = BELLE_preprocess('BelleGroup/train_2M_CN', tokenizer, max_length, 500000, require_enter=True)
        self.input_ids += input_ids
        self.labels += labels
        
        input_ids, labels = pCLUE_preprocess('wbbbbb/pclue', tokenizer, max_length, 300000)
        self.input_ids += input_ids
        self.labels += labels
                
    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])

def chat_preprocess(dataset, tokenizer: Callable, max_length: int = 512):
    input_ids = []
    output_lens = []

    for data in tqdm(dataset, disable=not is_rank_0(), mininterval=3):
        input_text = data['query'] + data['response'] + tokenizer.eos_token
        
        inputs = tokenizer(input_text,
                           max_length=max_length,
                           padding=False,
                           truncation=True,
                           return_tensors="pt")

        output_len = len(tokenizer.tokenize(data['response'] + tokenizer.eos_token))
        if output_len > max_length - 30:
            continue
        
        input_ids.append(inputs['input_ids'][0])
        output_lens.append(output_len)

    labels = copy.deepcopy(input_ids)
    for label, target_len in zip(labels, output_lens):
        label[:len(label) - target_len] = IGNORE_INDEX
    
    return input_ids, labels

class SFTDataset(Dataset):
    """
    Dataset for instruction tuning

    Args:
        tokenizer: tokenizer for supervised model
        data_path: InstructionWild data path
        max_length: max length of input
    """

    def __init__(self, tokenizer: Callable, max_length: int = 512, data_path: Optional[str] = None) -> None:
        super().__init__()
        # self.input_ids = []
        # self.labels = []
        
        dataset = jload(data_path)
        input_ids, labels = chat_preprocess(dataset, tokenizer, max_length)
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
