python inference.py \
    --model bloom \
    --model_path outputs/bloom-1b7-instruction-tuning \
    --input 'Hello' \
    --temperature 0.8 \
    --top_k 30 \
    --max_length 128 \
    --num_return_sequences 3
