#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=csc_8607
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --job-name=alexnet-full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

cd ~/TP3

echo "Début : $(date)"
echo "Nœud : $SLURM_NODELIST"
nvidia-smi

python -u train_imagenet.py \
    --model alexnet \
    --epochs 50 \
    --batch-size 32 \
    --workers 8

echo "Fin : $(date)"
