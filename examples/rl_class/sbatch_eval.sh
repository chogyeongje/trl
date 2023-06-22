#!/bin/bash

#SBATCH --job-name=trl_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/exp6.out
#SBATCH --partition=P1

source ${HOME}/.bashrc
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate torch

echo "DATE TIME : $(date +%Y)-$(date +%m)-$(date +%d) $(date +%H):$(date +%M):$(date +%S)"
echo "DIRECTORY : $PWD"

#srun python3 evaluate.py \
#	--model_name EXP6 \
#	--batch_size 32 

srun python3 make_baseline.py \
	--model_name "microsoft/DialoGPT-medium" \
	--batch_size 32 
