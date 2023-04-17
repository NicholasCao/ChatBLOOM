import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat(args, model, tokenizer, history):
    input_ids = tokenizer.encode(history, return_tensors='pt').to(torch.cuda.current_device())
    output = model.generate(input_ids,
        max_length=args.max_length,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        early_stopping=True)
    
    response = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bloom', choices=['gpt2', 'bloom', 'opt', 'roberta'])
    parser.add_argument('--model_path', type=str, default='outputs/bloom-1b7-sft-epoch3')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()
    
    
    if args.model == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.half()
    model.to(torch.cuda.current_device())
    model.eval()
    
    print('开始聊天。输入/reset清空聊天历史，输入/exit退出。')
    print('Start the chat. Type `/reset` to clear the chat history and `/exit` to exit.')
    history = ''
    while True:
        inp = input('Human: ').replace('\\n', '\n')
        if inp == '/exit':
            break
        if inp == '/reset':
            print('聊天历史已清空。Chat history is cleared.')
            history = ''
            continue

        history += f'Human: {inp}\nAssistant: '
        response = chat(args, model, tokenizer, history)
        print(f'Assistant: {response}')
    
    print('Bye ~ 👋')
