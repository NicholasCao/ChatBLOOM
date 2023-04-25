from typing import *
import re
import os
import json
import random
import argparse
from dataclasses import dataclass
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, GenerationConfig
from datasets import load_dataset
from colossalai.logging import get_dist_logger

from coati.dataset.utils import jload
from coati.dataset import DataCollatorForPromptDataset

logger = get_dist_logger()

CODE_KEYWORDS = ['python', 'java', 'c++', 'C#', 'javascript', 'php', 'golang']

def is_code_related(inp):
    inp = inp.lower()
    for key_word in CODE_KEYWORDS:
        if key_word in inp:
            return True
    return False

def get_filter_rate(res_len):
    if res_len < 5:
        return 0.97
    if res_len < 10:
        return 0.95
    if res_len < 20:
        return 0.9
    if res_len < 30:
        return 0.8
    if res_len < 50:
        return 0.7
    if res_len < 70:
        return 0.6
    if res_len < 80:
        return 0.5
    if res_len < 100:
        return 0.4
    return 0

def preprocess_instruct_dataset(dataset, filter: bool = False, tokenizer = None, start: int = 0, max_size: int = 10000):
    new_data = []
    
    for i, data in enumerate(dataset):
        if i < start:
            continue
        query = data['instruction'] + data['input']
        
        query = '<Human>: ' + query.strip().replace('\n\n', '\n') + '<eoh> <Assistant>: '
        response = data['output'].strip().replace('\n\n', '\n') + '<eoa>'
        
        if filter:
            # filter some short query
            if len(tokenizer.tokenize(query)) < 20:
                continue

            # filter some short response
            if random.random() < get_filter_rate(len(tokenizer.tokenize(response))):
                continue
            
            # filter some code related
            if is_code_related(query) and '```' not in response and random.random() < 0.5:
                continue
            
        new_data.append({'query': query, 'response': response})

        if len(new_data) >= max_size:
            break

    # construct context-independent conversations
    for i in range(0, len(new_data), 5):
        turn = random.randint(1, 3)
        new_query = ''
        
        for j in range(turn):
            index = random.randint(0, len(new_data) - 1)

            new_query += new_data[index]['query'] + new_data[index]['response']

        new_data[i] = {'query': new_query + new_data[i]['query'], 'response': new_data[i]['response']}
        
    return new_data

def preprocess_multiturn_chat(dataset, filter: bool = True, tokenizer = None, start: int = 0, max_size: int = 10000):
    new_data = []
    
    for i, data in enumerate(dataset):
        if i < start:
            continue

        query = data['instruction'].strip().replace('\n\n', '\n').replace('\nAssistant:', '<eoh> Assistant:')
        query = re.sub('Assistant:(?=\S+)', 'Assistant: ', query)
        query = query.replace('\nHuman:', '<eoa> Human:')
        query = re.sub('Human:(?=\S+)', 'Human: ', query)
        query = query.replace('Human:', '<Human>:').replace('Assistant:', '<Assistant>:') + ' '
        
        response = data['output'].strip().replace('\n\n', '\n') + '<eoa>'
        
        if filter:
            if len(tokenizer.tokenize(query)) < 50:
                continue

            if random.random() < get_filter_rate(len(tokenizer.tokenize(response))):
                continue
        new_data.append({'query': query, 'response': response})
        
        if len(new_data) >= max_size:
            break

    return new_data

def format_dialogue(text):
    text = text.strip().replace('\n\n', '\n')
    text = text.replace('\nAssistant:', '<eoh> Assistant:').replace('\nHuman:', '<eoa> Human:')
    text = text.replace('Human:', '<Human>:').replace('Assistant:', '<Assistant>:')
    return text + '<eoa>'

def generate(model, tokenizer, input_ids, attention_mask, generation_config: GenerationConfig):
    outputs = model.generate(input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config)
    
    response = [tokenizer.decode(output[len(inp_ids):], skip_special_tokens=True) for inp_ids, output in zip(input_ids, outputs)]
    return response

def chat_preprocess(dataset,
                    tokenizer: Callable,
                    max_length: int = 512,
                    max_size: int = 1000000,
                    start: int = 0):
    input_ids = []
    
    inputs, outputs = [], []

    for data in tqdm(dataset[start:]):
        inp = tokenizer(data['query'],
                           max_length=max_length,
                           padding=False,
                           truncation=True,
                           return_tensors="pt")

        input_ids.append(inp['input_ids'][0])
        
        inputs.append(data['query'])
        outputs.append(data['response'])
        
        if max_size is not None and len(input_ids) >= max_size:
            break
    
    return inputs, outputs, input_ids

class GenerateDataset(Dataset):

    def __init__(self, tokenizer: Callable, max_length: int = 512, data_path: Optional[str] = None) -> None:
        super().__init__()
        self.input_ids = []
        self.inputs = []
        self.outputs = []
        self.prompt_data = []
        
        belle_1M = load_dataset('BelleGroup/train_1M_CN', split='train')
        dataset = preprocess_instruct_dataset(belle_1M, filter=True, tokenizer=tokenizer, start=400000, max_size=30000)
        self.generate_prompt_data(dataset, tokenizer, start=0, max_size=8000)
        inputs, outputs, input_ids = chat_preprocess(dataset, tokenizer, max_length, max_size=20000, start=8000)
        self.inputs += inputs
        self.outputs += outputs
        self.input_ids += input_ids
        
        belle_mtchat = load_dataset('BelleGroup/multiturn_chat_0.8M', split='train')
        dataset = preprocess_multiturn_chat(belle_mtchat, filter=True, tokenizer=tokenizer, start=400000, max_size=30000)
        self.generate_prompt_data(dataset, tokenizer, start=0, max_size=8000)
        inputs, outputs, input_ids = chat_preprocess(dataset, tokenizer, max_length, max_size=20000, start=8000)
        self.inputs += inputs
        self.outputs += outputs
        self.input_ids += input_ids
        
        if data_path is not None:
            list_data_dict = jload(os.path.join(data_path, 'instinwild_ch.json'))
            list_data_dict_en = jload(os.path.join(data_path, 'instinwild_en.json'))
        
            dataset = preprocess_instruct_dataset(list_data_dict, filter=True, tokenizer=tokenizer, start=0, max_size=2000)
            self.generate_prompt_data(dataset, tokenizer, start=0, max_size=2000)
            inputs, outputs, input_ids = chat_preprocess(dataset, tokenizer, max_length, max_size=5000, start=10000)
            self.inputs += inputs
            self.outputs += outputs
            self.input_ids += input_ids
            
            dataset = preprocess_instruct_dataset(list_data_dict_en, filter=True, tokenizer=tokenizer, start=0, max_size=10000)
            self.generate_prompt_data(dataset, tokenizer, start=0, max_size=2000)
            inputs, outputs, input_ids = chat_preprocess(dataset, tokenizer, max_length, max_size=5000, start=10000)
            self.inputs += inputs
            self.outputs += outputs
            self.input_ids += input_ids
                
    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx])

    def generate_prompt_data(self, dataset, tokenizer, start: int = 0, max_size: int = 10000):
        raw_size = len(self.prompt_data)
        for data in tqdm(dataset[start:]):
            
            if len(tokenizer.tokenize(data['query'])) < 30 and random.random() < 0.4:
                continue
            self.prompt_data.append({
                'query': data['query'],
                'response': data['response']
            })
            if len(self.prompt_data) - raw_size >= max_size:
                break

    def save_prompt_data(self, path):
        with open(path, 'w') as file:
            json.dump(self.prompt_data, file, indent=2, sort_keys=True, ensure_ascii=False)

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
    
    tokenizer.padding_side = 'left'
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
    
    logger.info(f'Example: {data[0]}')

    dataset = GenerateDataset(tokenizer, args.max_prompt_length, data_path=args.data_path)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer)
    
    dataloader = DataLoader(dataset,
                sampler=SequentialSampler(dataset),
                batch_size=args.batch_size,
                collate_fn=data_collator,
                pin_memory=True)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        early_stopping=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    for batch_id, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(torch.cuda.current_device())
        attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
        
        responses = generate(model, tokenizer, input_ids, attention_mask, generation_config)
        
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

    logger.info(f'Example: {data[-1]}')

    with open(args.output_path, 'w') as file:
        json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False)
    
    logger.info(f'RM data size {len(data)}. Example: {data[0]}')
    logger.info(f'RM data generation completed, and saved to {args.output_path}.')
    
    # Generate prompt dataset
    logger.info(f'Prompt data size {len(dataset.prompt_data)}. Example: {dataset.prompt_data[0]}')
    dataset.save_prompt_data(args.prompt_output_path)
    logger.info(f'Prompt data generation completed, and saved to {args.prompt_output_path}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'roberta'])
    parser.add_argument('--model_path', type=str, default='outputs/bloom-1b7-sft')
    parser.add_argument('--max_prompt_length', type=int, default=384)
    parser.add_argument('--max_new_tokens', type=int, default=384)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--output_path', type=str, default='data/rm_data.json')
    parser.add_argument('--prompt_output_path', type=str, default='data/prompt_data.json')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    generate_data(args)
