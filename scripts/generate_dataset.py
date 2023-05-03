from typing import *
import re
import json
import random
import argparse
from tqdm import tqdm
import string

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

def is_zh(text):
    zh = r'[\u4e00-\u9fff]'
    result = re.findall(zh, text)
    return len(result) > 10

def is_english(text):
    cnt = 0
    for char in text:
        if char not in string.ascii_letters + string.punctuation + string.whitespace + string.digits:
            cnt += 1
            if cnt > min(len(text) * 0.02, 5):
                return False
    return True

def preprocess_sharegpt(dataset, tokenizer):
    new_data = []
    for data in tqdm(dataset):
        conversations = data['conversations']
        query = ''
        response = ''
        texts = []
        all_text = ''.join([conv['value'] for conv in conversations])
        
        if not is_zh(all_text) and not is_english(all_text):
            continue

        go_next = False
        for i, conv in enumerate(conversations):
            if conv['value'].lower().startswith('you:') or conv['value'].lower().startswith('pree:'):
                go_next = True
                break

            if conv['from'] in ['human', 'user']:
                # 0 from human
                texts.append(('<Human>:' + conv['value'].replace('\n\n', '\n'), 0))
            elif conv['from'] in ['gpt', 'chatgpt', 'bing', 'bard']:
                # 1 from ai
                texts.append(('<Assistant>:' + conv['value'].replace('\n\n', '\n'), 1))
            elif conv['from'] == 'system':
                continue
            else:
                raise NotImplementedError()

        if go_next or len(texts) < 2:
            continue
        
        if texts[-1][1] != 1:
            texts = texts[:-1]
            if len(texts) < 3 or (len(texts) == 2 and len(tokenizer.tokenze(texts[0][0])) < 20) :
                continue
            
        last = None
        go_next = False
        query = ''
        for text in texts[:-1]:
            if last is not None and last == text[1]:
                go_next = True
                break
            last = text[1]
            query += text[0]

        if go_next or texts[-1][1] == 0:
            continue

        query += '<Assistant>:'
        response = texts[-1][0].replace('<Assistant>:', '').strip()
    
        if 'gpt' not in response.lower() and 'openai' not in response.lower():
            new_data.append({'query': query, 'response': response})
        
    print(f'ShareGPT example: {new_data[0]}.')
    print(f'Number of examples: {len(new_data)}.')
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
    
    for data in tqdm(dataset):
        query = data['instruction'] + data['input']
        
        query = '<Human>:' + query.strip().replace('\n\n', '\n') + '<Assistant>:'
        response = data['output'].strip().replace('\n\n', '\n')
        
        if filter:
            # filter some short query
            if len(tokenizer.tokenize(query)) < 10:
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
        turn = random.randint(1, 3)
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
    
    for data in tqdm(dataset):

        query = data['instruction'].strip().replace('\n\n', '\n').replace('\nAssistant:', 'Assistant:')
        # query = re.sub('Assistant:(?=\S+)', 'Assistant: ', query)
        query = query.replace('\nHuman:', 'Human:')
        # query = re.sub('Human:(?=\S+)', 'Human: ', query)
        query = query.replace('Human:', '<Human>:').replace('Assistant:', '<Assistant>:')
        
        response = data['output'].strip().replace('\n\n', '\n')
        
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
    parser.add_argument('--sharegpt_data_path', type=str, default='data/sharegpt_20230401_clean_lang_split2.json')
    parser.add_argument('--gpt4llm_data_path', type=str, default='data/alpaca_gpt4_data.json')
    parser.add_argument('--gpt4llm_zh_data_path', type=str, default='data/alpaca_gpt4_data_zh.json')
    parser.add_argument('--sft_output_path', type=str, default='data/sft_data.json')
    parser.add_argument('--rm_output_path', type=str, default='data/rm_data.json')
    parser.add_argument('--prompt_output_path', type=str, default='data/prompt_data.json')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data = []

    dataset = jload(args.sharegpt_data_path)
    data += preprocess_sharegpt(dataset, tokenizer=tokenizer)

    dataset = jload(args.gpt4llm_data_path)
    data += preprocess_instruct_dataset(dataset)
    dataset = jload(args.gpt4llm_zh_data_path)
    data += preprocess_instruct_dataset(dataset)
    
    dataset = load_dataset('BelleGroup/train_1M_CN', split='train')
    data += preprocess_instruct_dataset(dataset, filter=True, tokenizer=tokenizer, max_size=120000)
    
    dataset = load_dataset('BelleGroup/multiturn_chat_0.8M', split='train')
    data += preprocess_multiturn_chat(dataset, filter=True, tokenizer=tokenizer, max_size=150000)
    
    print(f'Total data size: {len(data)}')

    random.shuffle(data)
    
    prompt_data = data[:20000]
    rm_data = data[20000:50000]
    sft_dta = data[50000:] + [{
        "query": "<Human>:你好<Assistant>:",
        "response": "您好！有什么可以帮助您的吗？"
    }, {
        "query": "<Human>:你是谁<Assistant>:",
        "response": "我是一个人工智能语言模型助手，可以进行自然语言交互，并尝试回答您的问题和提供帮助。"
    }]
    
    with open(args.sft_output_path, 'w') as file:
        json.dump(sft_dta, file)
    
    print(f'SFT data generation completed, and saved to {args.sft_output_path}.')
    
    with open(args.rm_output_path, 'w') as file:
        json.dump(rm_data, file, indent=2, ensure_ascii=False)
    
    print(f'RM data generation completed, and saved to {args.rm_output_path}.')
    
    with open(args.prompt_output_path, 'w') as file:
        json.dump(prompt_data, file, indent=2, ensure_ascii=False)

    print(f'Prompt data generation completed, and saved to {args.prompt_output_path}.')
