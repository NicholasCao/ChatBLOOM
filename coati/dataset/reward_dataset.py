from typing import *
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import transformers

from .utils import is_rank_0

class RMDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, special_token=None) -> None:
        super().__init__()
        self.chosen = []
        self.reject = []

        for data in tqdm(dataset, disable=not is_rank_0()):
            chosen = data['query'] + data['response'].strip()
            chosen_token = tokenizer(chosen,
                                     max_length=max_length,
                                     padding=False,
                                     truncation=True,
                                     return_tensors="pt")
            self.chosen.append({
                "input_ids": chosen_token['input_ids'][0],
            })

            reject = data['query'] + data['responses'][0].strip()
            reject_token = tokenizer(reject,
                                     max_length=max_length,
                                     padding=False,
                                     truncation=True,
                                     return_tensors="pt")
            self.reject.append({
                "input_ids": reject_token['input_ids'][0],
            })

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return dict(chosen_input_ids=self.chosen[idx]["input_ids"], reject_input_ids=self.reject[idx]["input_ids"])

@dataclass
class DataCollatorForRMDataset(object):
    """Collate examples for reward model."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, reject_input_ids = tuple([instance[key] for instance in instances] for key in ("chosen_input_ids", "reject_input_ids"))
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(chosen_input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        reject_input_ids = torch.nn.utils.rnn.pad_sequence(reject_input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        return dict(
            chosen_input_ids=chosen_input_ids,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            reject_input_ids=reject_input_ids,
            reject_attention_mask=reject_input_ids.ne(self.tokenizer.pad_token_id),
        )
