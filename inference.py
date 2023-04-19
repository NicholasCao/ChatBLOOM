import argparse

import torch
from coati.models.bloom import BLOOMActor
from coati.models.gpt import GPTActor
from coati.models.opt import OPTActor
from coati.models.roberta import RoBERTaActor
from transformers import AutoTokenizer, RobertaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from transformers import AutoModelForCausalLM

def eval(args):
    # configure model
    if args.model == 'gpt2':
        actor = GPTActor(pretrained=args.model_path).to(torch.cuda.current_device())
    elif args.model == 'bloom':
        actor = BLOOMActor(pretrained=args.model_path).to(torch.cuda.current_device())
    elif args.model == 'opt':
        actor = OPTActor(pretrained=args.model_path).to(torch.cuda.current_device())
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    actor.eval()
    input = args.input
    input_ids = tokenizer.encode(input, return_tensors='pt').to(torch.cuda.current_device())
    # outputs = actor.generate(input_ids,
    #                          max_length=args.max_length,
    #                          do_sample=True,
    #                          temperature=args.temperature,
    #                          top_k=args.top_k,
    #                          top_p=args.top_p,
    #                          num_return_sequences=1,
    #                          eos_token_id=tokenizer.eos_token_id,
    #                          pad_token_id=tokenizer.pad_token_id,
    #                          early_stopping=True)
    # output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(torch.cuda.current_device())
    model.eval()
    outputs = model.generate(input_ids,
        max_length=args.max_length,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        early_stopping=True,
        no_repeat_ngram_size=1)
    # output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('Input:\n' + 100 * '-')
    print('\033[96m' + args.input + '\033[0m')
    print('\nOutput:\n' + 100 * '-')
    for i in range(len(outputs)):
        print('\033[92m' + tokenizer.decode(outputs[i], skip_special_tokens=True) + '\033[0m\n')
    # output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'roberta'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--input', type=str, default='Question: How are you ? Answer:')
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_return_sequences', type=int, default=3)
    args = parser.parse_args()
    eval(args)
