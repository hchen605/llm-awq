#!/bin/bash
#SBATCH -J job_id
#SBATCH -o ./log/pip-list.out
#SBATCH --gres=gpu:1 #Number of GPU devices to use [0-2]
#SBATCH --nodelist=leon05 #YOUR NODE OF PREFERENCE

# Set Hugging Face token
#export CUDA_VISIBLE_DEVICES=0,1

module load shared singularity 

#singularity exec --nv ../img/llm-awq.img pip list

# Run AWQ latency/ttft
singularity exec --nv ../img/llm-awq.img \
    python demo.py --model_type llama \
    --model_path ../../Meta-Llama-3-8B \
    --q_group_size 128 --load_quant ../quant_cache/llama3-8b-w4-g128-awq-v2.pt \
    --precision W4A16 \
    --prompt "USER: hi, there! how are you doing? My name is Hsin. I would like to ask about the price of health care plans. Do you know any related information?"

# run raw model
# singularity exec --nv ../img/llm-awq.img \
#     python demo.py --model_type llama \
#     --model_path ../../Meta-Llama-3-8B \
#     --precision W16A16 \
#     --prompt "USER: hi, there! how are you doing? My name is Hsin. I would like to ask about the price of health care plans. Do you know any related information?"