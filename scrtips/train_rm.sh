# CUDA_VISIBLE_DEVICES=2,3 
torchrun --standalone --nproc_per_node=4 train_rm.py \
    --pretrain 'bigscience/bloom-1b1' \
    --model 'bloom' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_sig'\
    --save_path  outputs/bloom-1b1-rm \
    --batch_size 4 \
    --accimulation_steps 2 \
    --max_epochs 2 \
    --lr 5e-6

torchrun --standalone --nproc_per_node=4 train_rm.py \
    --pretrain 'bigscience/bloom-1b1' \
    --model 'bloom' \
    --strategy colossalai_zero2 \
    --loss_fn 'log_exp'\
    --save_path outputs/bloom-1b1-rm_exp \
    --batch_size 4 \
    --accimulation_steps 2 \
    --max_epochs 2 \
    --lr 5e-6
