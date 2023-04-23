torchrun --standalone --nproc_per_node=4 train_sft.py \
    --pretrain outputs/bloom-1b7-instruction-tuning \
    --model bloom \
    --strategy colossalai_zero2 \
    --data_path data/sft_data.json \
    --save_path  outputs/bloom-1b7-sft \
    --batch_size 4 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_len 768 \
    --max_epochs 3

python generate_rm_dataset.py --model_path outputs/bloom-1b7-sft

torchrun --standalone --nproc_per_node=4 train_rm.py \
    --pretrain 'bigscience/bloom-560m' \
    --model 'bloom' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_sig'\
    --save_path  outputs/bloom-560m-rm \
    --batch_size 8 \
    --accumulation_steps 1 \
    --max_epochs 1 \
    --lr 1e-5 \
    --data_path data/rm_data.json

