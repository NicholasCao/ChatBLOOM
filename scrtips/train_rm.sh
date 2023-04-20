# torchrun --standalone --nproc_per_node=4 train_rm.py \
#     --pretrain 'bigscience/bloom-1b1' \
#     --model 'bloom' \
#     --strategy colossalai_zero2 \
#     --loss_fn 'log_sig'\
#     --save_path  outputs/bloom-1b1-rm \
#     --batch_size 4 \
#     --accimulation_steps 2 \
#     --max_epochs 2 \
#     --lr 5e-6

torchrun --standalone --nproc_per_node=4 train_rm.py \
    --pretrain 'bigscience/bloom-560m' \
    --model 'bloom' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_sig'\
    --save_path  outputs/bloom-560m-rm-1 \
    --batch_size 8 \
    --accimulation_steps 1 \
    --max_epochs 1 \
    --lr 5e-6

# torchrun --standalone --nproc_per_node=4 train_rm.py \
#     --pretrain 'bigscience/bloom-560m' \
#     --model 'bloom' \
#     --strategy colossalai_zero2 \
#     --loss_fn 'log_sig'\
#     --save_path  outputs/bloom-560m-rm-2 \
#     --batch_size 8 \
#     --accimulation_steps 1 \
#     --max_epochs 2 \
#     --lr 5e-6
