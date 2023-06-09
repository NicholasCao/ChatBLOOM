import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def chat(args, model, tokenizer, history):
    inputs = tokenizer(history, return_tensors='pt', max_length=args.prompt_max_length, truncation=True).to(model.device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    output = model.generate(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            max_length=args.max_length,
                            do_sample=True,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            repetition_penalty=1.05,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='nicholascao/chatbloom-1b7-sft')
    parser.add_argument('--prompt_max_length', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.half()
    model.to(torch.cuda.current_device())
    model.eval()

    print('开始聊天。输入/reset清空聊天历史，输入/exit退出。')
    print('Start the chat. Type `/reset` to clear the chat history and `/exit` to exit.')
    history = ''
    while True:
        inp = input('<Human>: ').replace('\\n', '\n')
        if inp == '/exit':
            break
        if inp == '/reset':
            print('聊天历史已清空。Chat history is cleared.')
            history = ''
            continue

        history += f'<Human>:{inp}<Assistant>:'
        response = chat(args, model, tokenizer, history)
        history += response
        print(f'<Assistant>: {response}')
    
    print('Bye ~ 👋')
