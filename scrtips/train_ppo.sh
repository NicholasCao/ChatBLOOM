# set_n_least_used_CUDA_VISIBLE_DEVICES() {
#     local n=${1:-"9999"}
#     echo "GPU Memory Usage:"
#     local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
#         | tail -n +2 \
#         | nl -v 0 \
#         | tee /dev/tty \
#         | sort -g -k 2 \
#         | awk '{print $1}' \
#         | head -n $n)
#     export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
#     echo "Now CUDA_VISIBLE_DEVICES is set to:"
#     echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# }

# set_n_least_used_CUDA_VISIBLE_DEVICES 2

torchrun --standalone --nproc_per_node=4 train_ppo.py \
    --prompt_path /path/to/data.json \
    --strategy colossalai_zero2 \
    --model bloom \
    --pretrain outputs/bloom-1b7-sft-epoch5 \
    --rm_model bloom \
    --rm_pretrain bigscience/bloom-560m \
    --rm_path outputs/bloom-560m-rm-1 \
    --lr 3e-6 \
    --instruction_max_length 256 \
    --max_length 512 \
    --ptx_coef 0 \
    --train_batch_size 4 \
    --accimulation_steps 2 \
    --save_path outputs/bloom-1b7_ppo \
    --prompt_path data/rm_data.json \
    --num_episodes 1 \
    --update_timesteps 30 \
    --max_epochs 3
    
    #  \
    # --max_datasets_size 10000
