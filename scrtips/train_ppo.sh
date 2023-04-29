
torchrun --standalone --nproc_per_node=4 train_ppo.py \
    --prompt_path data/prompt_data.json \
    --strategy colossalai_zero2 \
    --model bloom \
    --pretrain outputs/bloom-1b7-sft \
    --rm_model bloom \
    --rm_pretrain outputs/bloom-560m-rm \
    --lr 1e-6 \
    --prompt_max_length 448 \
    --max_length 768 \
    --batch_size 32 \
    --mini_batch_size 2 \
    --accumulation_steps 4 \
    --save_path outputs/bloom-1b7-ppo \
    --ppo_epochs 4 \
    --kl_coef 0.05 \
    --vf_coef 0.1 \
    --reward_baseline 20 \
    --freeze_layer_ratio 0.67
