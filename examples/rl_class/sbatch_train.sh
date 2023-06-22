#!/bin/bash

#SBATCH --job-name=trl
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/EXP3.out
#SBATCH --partition=P1

source ${HOME}/.bashrc
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate torch

echo "DATE TIME : $(date +%Y)-$(date +%m)-$(date +%d) $(date +%H):$(date +%M):$(date +%S)"
echo "DIRECTORY : $PWD"

ACCELERATE_LOG_LEVEL=info 
srun accelerate launch \
	rl_class.py \
	--batch_size 32 \
	--mini_batch_size 16 \
	--ppo_epoch 100 \
	--learning_rate 5e-6 \
	--model_save_path EXP3 \
	--use_usefulness True \
	--use_harmfulness False \
	--lambda_type linear \
	--lambda_value -1 \
	--lambda_lr 0.1 \
	--max_constraint 1

