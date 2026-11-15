#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH -J tp6-ex5
#SBATCH -o logs/%x-%j.out

set -euo pipefail

source tp6_env.sh

# Nécessaire pour les noyaux INT8 x86 lors de la compilation
export TORCHINDUCTOR_FREEZING=1

CKPT=best.pth
[ -f "$CKPT" ] || CKPT=/mnt/hdd/homes/jromero/tp6/best_resnet50_imagenet10.pth

# Expérience 1 : quatre variantes, calibration sur 200 images
python -u quantize.py --tag calib200 --ckpt "$CKPT" --threads 4 --calib 200

# Expérience 2 : INT8 uniquement, calibration sur 10 images
python -u quantize.py --tag calib10 --ckpt "$CKPT" --threads 4 --calib 10 --variants int8

# Expérience 3 : comparaison avec un seul thread CPU
python -u quantize.py --tag threads1 --ckpt "$CKPT" --threads 1 --variants fp32,int8 --n-val 100

