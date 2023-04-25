import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

from colossalai.logging import get_dist_logger

from .utils import is_rank_0, jload

logger = get_dist_logger()


class PromptDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: int = 256, max_datasets_size: int = None):
        super(PromptDataset, self).__init__()
        self.prompt = []
        logger.info("Loading data...", ranks=[0])
        list_data_dict = jload(data_path)
        logger.info(f"Loaded {len(list_data_dict)} examples.", ranks=[0])

        if max_datasets_size is not None:
            logger.info(f"Limiting dataset to {max_datasets_size} examples.", ranks=[0])
            random.shuffle(list_data_dict)
            list_data_dict = list_data_dict[:max_datasets_size]

        for i, data_dict in enumerate(list_data_dict):
            text = data_dict['query']

            if len(tokenizer.tokenize(text)) > max_length and random.random() < 0.3:
                continue

            token = tokenizer(text,
                              return_tensors='pt',
                              max_length=max_length,
                              padding=False,
                              truncation=True)

            self.prompt.append(token['input_ids'][0].to(torch.cuda.current_device()))

            if max_datasets_size is not None and len(self.prompt) >= max_datasets_size:
                break

    def __len__(self):
        return len(self.prompt)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.prompt[i])

@dataclass
class DataCollatorForPromptDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids = [instance['input_ids'] for instance in instances]
        batch_mask = [torch.ones_like(element) for element in input_ids]
        inputs = {"input_ids": input_ids, "attention_mask": batch_mask}

        padded_inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=None,
            return_tensors="pt",
        )
        return dict(
            input_ids=padded_inputs['input_ids'],
            attention_mask=padded_inputs['attention_mask'],
        )
