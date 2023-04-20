
torchrun --standalone --nproc_per_node=4 train_sft.py \
    --pretrain outputs/bloom-1b7-instruction-tuning \
    --model bloom \
    --strategy colossalai_zero2 \
    --data_path data \
    --save_path  outputs/bloom-1b7-sft-epoch5_2 \
    --batch_size 4 \
    --accimulation_steps 8 \
    --lr 3e-5 \
    --max_len 512 \
    --max_epochs 5

# use bloomz
# torchrun --standalone --nproc_per_node=4 train_sft.py \
#     --pretrain bigscience/bloomz-1b7 \
#     --model bloom \
#     --strategy colossalai_zero2 \
#     --data_path data \
#     --save_path  outputs/bloomz-1b7-sft-epoch5 \
#     --batch_size 4 \
#     --accimulation_steps 8 \
#     --lr 2e-5 \
#     --max_len 512 \
#     --max_epochs 5
