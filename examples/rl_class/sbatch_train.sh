#!/bin/bash

#SBATCH --job-name=trl
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm.out
#SBATCH --partition=3090

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
	--ppo_epoch 5 \
	--model_save_path test \
	--use_usefulness True \
	--use_harmfulness False
