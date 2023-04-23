from typing import *
import re
import json
import random
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset

from coati.dataset.utils import jload

CODE_KEYWORDS = ['python', 'java', 'c++', 'C#', 'javascript', 'php', 'golang']

def is_code_related(inp):
    inp = inp.lower()
    for key_word in CODE_KEYWORDS:
        if key_word in inp:
            return True
    return False

def preprocess_sharegpt(dataset):
    new_data = []
    for data in dataset:
        conversations = data['conversations']
        query = ''
        response = ''
        for i, conv in enumerate(conversations):
            text = ''
            if conv['from'] in ['human', 'user']:
                text += '<<Human>: ' + conv['value'].replace('\n\n', '\n') + '<eoh>'
            elif conv['from'] in ['gpt', 'chatgpt', 'bing', 'bard']:
                text += '<<Assistant>: ' + conv['value'].replace('\n\n', '\n') + '<eoa>'
            elif conv['from'] == 'system':
                continue
            else:
                print(conv['from'])
                raise NotImplementedError()
                
            if i == len(conversations) - 1:
                response += text
            else:
                query += text
    
        if 'gpt' not in response.lower():
            new_data.append({'query': query, 'response': response})
        
    print(f'ShareGPT example: {new_data[0]}. Number of examples: {len(new_data)}')
    
    return new_data

def get_filter_rate(res_len):
    if res_len < 5:
        return 0.95
    if res_len < 10:
        return 0.9
    if res_len < 20:
        return 0.8
    if res_len < 30:
        return 0.7
    if res_len < 40:
        return 0.6
    if res_len < 50:
        return 0.5
    if res_len < 60:
        return 0.3
    return 0

def preprocess_instruct_dataset(dataset, filter: bool = False, tokenizer = None, max_size: int = -1):
    new_data = []
    
    for data in dataset:
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

        if max_size > 0 and len(new_data) >= max_size * 2:
            break
            
        new_data.append({'query': query, 'response': response})

    # construct context-independent conversations
    for i in range(0, len(new_data), 5):
        turn = random.randint(1, 4)
        new_query = ''
        
        for j in range(turn):
            if max_size > 0:
                index = random.randint(max_size, len(new_data) - 1)
            else:
                index = random.randint(0, len(new_data) - 1)

            new_query += new_data[index]['query'] + new_data[index]['response']

        new_data[i] = {'query': new_query + new_data[i]['query'], 'response': new_data[i]['response']}
    
    if max_size > 0:
        new_data = new_data[:max_size]
    
    print(f'Instruction example: {new_data[0]}.')
    print(f'Instruction example: {new_data[1]}.')
    print(f'Number of examples: {len(new_data)}.')
        
    return new_data

def preprocess_multiturn_chat(dataset, filter: bool = True, tokenizer = None, max_size: int = -1):
    new_data = []
    
    for data in dataset:

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

        if max_size > 0 and len(new_data) >= max_size:
            break

        new_data.append({'query': query, 'response': response})

    print(f'MultiturnChat example: {new_data[0]}.')
    print(f'MultiturnChat example: {new_data[1]}.')
    print(f'Number of examples: {len(new_data)}.')
        
    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='bigscience/bloom-1b7')
    parser.add_argument('--sharegpt_data_path', type=str, default='data/sharegpt_20230401_clean_lang_split.json')
    parser.add_argument('--gpt4llm_data_path', type=str, default='data/alpaca_gpt4_data.json')
    parser.add_argument('--gpt4llm_zh_data_path', type=str, default='data/alpaca_gpt4_data_zh.json')
    parser.add_argument('--output_path', type=str, default='data/sft_data.json')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data = []

    dataset = jload(args.sharegpt_data_path)
    data += preprocess_sharegpt(dataset)

    dataset = jload(args.gpt4llm_data_path)
    data += preprocess_instruct_dataset(dataset)
    dataset = jload(args.gpt4llm_zh_data_path)
    data += preprocess_instruct_dataset(dataset)
    
    dataset = load_dataset('BelleGroup/train_1M_CN', split='train')
    data += preprocess_instruct_dataset(dataset, filter=True, tokenizer=tokenizer, max_size=80000)
    
    dataset = load_dataset('BelleGroup/multiturn_chat_0.8M', split='train')
    data += preprocess_multiturn_chat(dataset, filter=True, tokenizer=tokenizer, max_size=100000)
    
    print(f'Total data size: {len(data)}')

    with open(args.output_path, 'w') as file:
        json.dump(data, file)
    
    print(f'SFT data generation completed, and saved to {args.output_path}.')
