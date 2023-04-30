from typing import *
import json
import argparse
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, GenerationConfig
from colossalai.logging import get_dist_logger

from coati.dataset.utils import jload
from coati.dataset import DataCollatorForPromptDataset

logger = get_dist_logger()

@torch.no_grad()
def generate(args, model, tokenizer, input_ids, attention_mask):
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=args.max_length,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        early_stopping=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = [tokenizer.decode(output[len(inp_ids):], skip_special_tokens=True) for inp_ids, output in zip(input_ids, outputs)]
    return response

class GenerateDataset(Dataset):
    def __init__(self, dataset, tokenizer: Callable, max_length: int, num_return_sequences: int) -> None:
        super().__init__()
        self.querys = []
        self.responses = []
        self.input_ids = []
        for data in tqdm(dataset):
            inp = tokenizer(data['query'],
                            max_length=max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt")

            self.querys.extend([data['query']] * num_return_sequences)
            self.responses.extend([data['response']] * num_return_sequences)
            self.input_ids.extend([inp['input_ids'][0]] * num_return_sequences)

    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx])

def generate_data(args):
    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.half()

    model.to(torch.cuda.current_device())
    model.eval()
    
    dataset = jload(args.data_path)

    if args.end is not None:
        dataset = dataset[args.start:args.end]
    else:
        dataset = dataset[args.start:]
    
    dataset = GenerateDataset(dataset, tokenizer, args.prompt_max_length, num_return_sequences=args.num_return_sequences)
    data_collator = DataCollatorForPromptDataset(tokenizer=tokenizer)
    
    dataloader = DataLoader(dataset,
                sampler=SequentialSampler(dataset),
                batch_size=args.batch_size,
                collate_fn=data_collator,
                pin_memory=True)

    new_rm_data = []

    # generate
    for batch_id, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(torch.cuda.current_device())
        attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
        
        responses = generate(args, model, tokenizer, input_ids, attention_mask)
        
        for i in range(len(responses) // args.num_return_sequences):
            qeury = dataset.querys[batch_id * args.batch_size + i * args.num_return_sequences]
            response = dataset.responses[batch_id * args.batch_size + i * args.num_return_sequences]
            new_rm_data.append({
                'query': qeury,
                'response': response,
                'responses': responses[i * args.num_return_sequences: (i + 1) * args.num_return_sequences]
            })
        
        if (batch_id + 1) % 200 == 0:
            # save rm data
            with open(args.output_path, 'w') as file:
                json.dump(new_rm_data, file, indent=2, ensure_ascii=False)

    # save rm data
    with open(args.output_path, 'w') as file:
        json.dump(new_rm_data, file, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='outputs/bloom-1b7-sft')
    parser.add_argument('--prompt_max_length', type=int, default=448)
    parser.add_argument('--max_length', type=int, default=768)
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--data_path', type=str, default='data/rm_data.json')
    parser.add_argument('--output_path', type=str, default='data/generated_rm_data.json')
    parser.add_argument('--num_return_sequences', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    assert args.batch_size % args.num_return_sequences == 0

    generate_data(args)
