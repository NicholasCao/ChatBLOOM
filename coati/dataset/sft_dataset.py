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

CODE_KEYWORDS = ['python', 'java', 'c++', 'C#', 'javascript', 'php', 'golang']

def is_code_related(inp):
    inp = inp.lower()
    for key_word in CODE_KEYWORDS:
        if key_word in inp:
            return True
    return False

def get_filter_rate(res_len):
    if res_len < 5:
        return 0.9
    if res_len < 10:
        return 0.8
    if res_len < 20:
        return 0.6
    if res_len < 40:
        return 0.4
    if res_len < 50:
        return 0.3
    if res_len < 60:
        return 0.2
    if res_len < 70:
        return 0.1
    return 0

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

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def format_chat(dataset, input_key: str = 'instruction', output_key: str = 'output', is_chat: bool = False):
    inputs = []
    outputs = []

    if is_chat:
        for data in dataset:
            formated_input = data[input_key].strip().replace('\n\n', '\n').replace('\nAssistant:', '<eoh> Assistant:')
            formated_input = re.sub('Assistant:(?=\S+)', 'Assistant: ', formated_input)
            formated_input = formated_input.replace('\nHuman:', '<eoa> Human:')
            formated_input = re.sub('Human:(?=\S+)', 'Human: ', formated_input)
            formated_input = formated_input.replace('Human:', '<Human>:').replace('Assistant:', '<Assistant>:') + ' '
            
            formated_output = data[output_key].strip().replace('\n\n', '\n') + '<eoa>'
            inputs.append(formated_input)
            outputs.append(formated_output)
    else:
        for data in dataset:
            formated_input = '<Human>: ' + data[input_key].strip().replace('\n\n', '\n') + '<eoh> <Assistant>: '
            formated_output = data[output_key].strip().replace('\n\n', '\n') + '<eoa>'
            inputs.append(formated_input)
            outputs.append(formated_output)

        # construct context-independent conversations
        for i in range(0, len(inputs), 4):
            turn = random.randint(1, 5)
            new_inp = ''
            new_oup = ''
            
            for j in range(turn):
                index = random.randint(0, len(inputs) - 1)
                inp = inputs[index]
                oup = outputs[index]

                new_inp += inp + oup

            new_inp += inputs[i]
            new_oup += outputs[i]
            
            inputs[i] = new_inp
            outputs[i] = new_oup

    logger.info(f'Example: \n{inputs[0]}{outputs[0]}', ranks=[0])
    logger.info(f'Example: \n{inputs[1]}{outputs[1]}', ranks=[0])
    return inputs, outputs
    

def chat_preprocess(inputs: List[str],
                    outputs: List[str],
                    tokenizer: Callable,
                    max_length: int = 512,
                    max_datasets_size: int = 1000000,
                    short_text_len: int = 50,
                    start: int = 0):
    input_ids = []
    input_lens = []
    short_response_count = 0
    count = 0

    for inp, oup in tqdm(zip(inputs[start:], outputs[start:]), disable=not is_rank_0(), mininterval=3):
        count += 1

        # filter some code related
        if is_code_related(inp) and '```' not in oup and random.random() < 0.5:
            continue

        # filter some short query
        if len(tokenizer.tokenize(inp)) < 20 and random.random() < 0.3:
            continue

        # filter some short response
        res_len = len(tokenizer.tokenize(oup))
        if res_len < 70:
            rate = get_filter_rate(res_len)
            if rate > 0 and random.random() < get_filter_rate(res_len):
                continue
            if res_len < short_text_len and short_response_count / max_datasets_size > 0.25:
                continue
            if res_len < short_text_len:
                short_response_count += 1

        input_text = inp + oup + tokenizer.eos_token
        
        inputs = tokenizer(input_text,
                           max_length=max_length,
                           padding=False,
                           truncation=True,
                           return_tensors="pt")

        if len(inputs['input_ids'][0]) < 20:
            continue
        
        input_len = len(tokenizer.tokenize(inp))
        
        input_ids.append(inputs['input_ids'][0])
        input_lens.append(input_len)
        
        if max_datasets_size is not None and len(input_ids) >= max_datasets_size:
            break

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, input_lens):
        if random.random() < 0.95:
            label[:source_len] = IGNORE_INDEX
    
    logger.info(f"The data set is enumerated to {count}, short response rate = {short_response_count / len(input_ids)}", ranks=[0])
    
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
        self.input_ids = []
        self.labels = []
        
        belle_1M = load_dataset('BelleGroup/train_1M_CN', split='train')
        inputs, outputs = format_chat(belle_1M, "instruction", "output")
        input_ids, labels = chat_preprocess(inputs, outputs, tokenizer, max_length, max_datasets_size=240000, short_text_len=60)
        self.input_ids += input_ids
        self.labels += labels
        
        belle_mtchat = load_dataset('BelleGroup/multiturn_chat_0.8M', split='train')
        inputs, outputs = format_chat(belle_mtchat, "instruction", "output", is_chat=True)
        input_ids, labels = chat_preprocess(inputs, outputs, tokenizer, max_length, max_datasets_size=200000, short_text_len=50)
        self.input_ids += input_ids
        self.labels += labels
        
        if data_path is not None:
            list_data_dict = jload(os.path.join(data_path, 'instinwild_ch.json'))
            list_data_dict_en = jload(os.path.join(data_path, 'instinwild_en.json'))
        
            inputs, outputs = format_chat(list_data_dict, "instruction", "output")
            input_ids, labels = chat_preprocess(inputs, outputs, tokenizer, max_length, max_datasets_size=30000, short_text_len=20)
            self.input_ids += input_ids
            self.labels += labels
            
            inputs, outputs = format_chat(list_data_dict_en, "instruction", "output")
            input_ids, labels = chat_preprocess(inputs, outputs, tokenizer, max_length, max_datasets_size=30000, short_text_len=20)
            self.input_ids += input_ids
            self.labels += labels
                
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
