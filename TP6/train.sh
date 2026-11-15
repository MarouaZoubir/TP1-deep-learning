#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH -J tp6-train
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source tp6_env.sh

python -u train.py \
    --tag best-bf16-cl-compile \
    --epochs 2 \
    --batch-size 64 \
    --lr 3e-4 \
    --precision bf16 \
    --channels-last \
    --compile \
    --data small \
    --workers 3 \
    --out best.pth
