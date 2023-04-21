from typing import *
import os
import json
import random
import argparse
from dataclasses import dataclass
import torch

from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from datasets import load_dataset

from coati.dataset.sft_dataset import format_chat
from coati.dataset.utils import jload

from tqdm import tqdm

def format_dialogue(text):
    text = text.strip().replace('\n\n', '\n')
    text = text.replace('\nAssistant:', '<eoh> Assistant:').replace('\nHuman:', '<eoa> Human:')
    text = text.replace('Human:', '<Human>:').replace('Assistant:', '<Assistant>:')
    return text + '<eoa>'

@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def generate(args, model, tokenizer, input_ids, attention_mask):
    outputs = model.generate(input_ids,
        attention_mask=attention_mask,
        max_length=args.max_length,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        early_stopping=True)
    
    response = [tokenizer.decode(output[len(inp_ids):], skip_special_tokens=True) for inp_ids, output in zip(input_ids, outputs)]
    return response

def chat_preprocess(inputs: List[str],
                    outputs: List[str],
                    tokenizer: Callable,
                    max_length: int = 512,
                    max_datasets_size: int = 1000000,
                    start: int = 0):
    input_ids = []
    input_lens = []
    count = 0

    tokenizer.truncation_side = 'left'
    
    filtered_inputs, filtered_outputs = [], []

    for inp, oup in zip(inputs[start:], outputs[start:]):
        count += 1
        
        # filter some short query
        if len(tokenizer.tokenize(inp)) < 20 and random.random() < 0.3:
            continue
    
        # filter some short response
        if len(tokenizer.tokenize(oup)) < 30 and random.random() < 0.3:
            continue

        input_text = inp #+ oup + tokenizer.eos_token
        
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
        
        filtered_inputs.append(inp)
        filtered_outputs.append(oup)
        
        if max_datasets_size is not None and len(input_ids) >= max_datasets_size:
            break
    
    print(f"The data set is enumerated to {count}.")
    
    return filtered_inputs, filtered_outputs, input_ids

class RMGenerateDataset(Dataset):

    def __init__(self, tokenizer: Callable, max_length: int = 512, data_path: Optional[str] = None) -> None:
        super().__init__()
        self.input_ids = []
        self.inputs = []
        self.outputs = []
        
        belle_1M = load_dataset('BelleGroup/train_1M_CN', split='train')
        inputs, outputs = format_chat(belle_1M, "instruction", "output")
        inputs, outputs, input_ids = chat_preprocess(inputs, outputs, tokenizer, max_length, max_datasets_size=20000, start=900000)
        self.inputs += inputs
        self.outputs += outputs
        self.input_ids += input_ids
        
        belle_mtchat = load_dataset('BelleGroup/multiturn_chat_0.8M', split='train')
        inputs, outputs = format_chat(belle_mtchat, "instruction", "output", is_chat=True)
        inputs, outputs, input_ids = chat_preprocess(inputs, outputs, tokenizer, max_length, max_datasets_size=20000, start=700000)
        self.inputs += inputs
        self.outputs += outputs
        self.input_ids += input_ids
        
        if data_path is not None:
            list_data_dict = jload(os.path.join(data_path, 'instinwild_ch.json'))
            list_data_dict_en = jload(os.path.join(data_path, 'instinwild_en.json'))
        
            inputs, outputs = format_chat(list_data_dict, "instruction", "output")
            inputs, outputs, input_ids = chat_preprocess(inputs, outputs, tokenizer, max_length, max_datasets_size=10000, short_text_len=30, start=0)
            self.inputs += inputs
            self.outputs += outputs
            self.input_ids += input_ids
            
            inputs, outputs = format_chat(list_data_dict_en, "instruction", "output")
            inputs, outputs, input_ids = chat_preprocess(inputs, outputs, tokenizer, max_length, max_datasets_size=10000, short_text_len=30, start=0)
            self.inputs += inputs
            self.outputs += outputs
            self.input_ids += input_ids
                
    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx])


def generate_data(args):
    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    
    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.half()

    model.to(torch.cuda.current_device())
    model.eval()

    data = []
    
    harmless_datas = load_dataset('Anthropic/hh-rlhf', data_dir="harmless-base", split='train')
    for i, instance in enumerate(harmless_datas):
        if i >= 5000:
            break
        data.append({
            "chosen": format_dialogue(instance['chosen']),
            "rejected": format_dialogue(instance['rejected'])
        })
    helpful_datas = load_dataset('Anthropic/hh-rlhf', data_dir="helpful-base", split='train')
    for i, instance in enumerate(helpful_datas):
        if i >= 5000:
            break
        data.append({
            "chosen": format_dialogue(instance['chosen']),
            "rejected": format_dialogue(instance['rejected'])
        })

    dataset = RMGenerateDataset(tokenizer, args.max_length, data_path=args.data_path)
    data_collator = DataCollator(tokenizer=tokenizer)
    
    dataloader = DataLoader(dataset,
                sampler=SequentialSampler(dataset),
                batch_size=args.batch_size,
                collate_fn=data_collator,
                pin_memory=True)
    
    for batch_id, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(torch.cuda.current_device())
        attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
        
        responses = generate(args, model, tokenizer, input_ids, attention_mask)
        for inp, oup, response in zip(
            dataset.inputs[batch_id*args.batch_size:(batch_id+1)*args.batch_size],
            dataset.outputs[batch_id*args.batch_size:(batch_id+1)*args.batch_size],
            responses):
            chosen = inp + oup
            rejected = inp + response
            data.append({
                "chosen": chosen,
                "rejected": rejected
            })

    with open(args.output_path, 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'roberta'])
    parser.add_argument('--model_path', type=str, default='outputs/bloom-1b7-sft-epoch5')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='data/rm_data.json')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    generate_data(args)
