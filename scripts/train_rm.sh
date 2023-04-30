# torchrun --standalone --nproc_per_node=4 train_rm.py \
#     --pretrain 'bigscience/bloom-560m' \
#     --model 'bloom' \
#     --strategy colossalai_zero2 \
#     --loss_fn 'log_sig'\
#     --save_path  outputs/bloom-560m-rm \
#     --batch_size 8 \
#     --accumulation_steps 1 \
#     --max_epochs 1 \
#     --lr 5e-6 \
#     --data_path data/generated_rm_data.json \
#     --max_len 768 \
#     --freeze_layer_ratio 0.33

torchrun --standalone --nproc_per_node=4 train_rm.py \
    --pretrain 'bigscience/bloom-1b1' \
    --model 'bloom' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_sig'\
    --save_path  outputs/bloom-1b7-rm \
    --batch_size 4 \
    --accumulation_steps 2 \
    --max_epochs 1 \
    --lr 5e-6 \
    --data_path data/generated_rm_data.json \
    --max_len 768 \
    --freeze_layer_ratio 0.33