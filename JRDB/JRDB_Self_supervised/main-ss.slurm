#!/bin/bash
#SBATCH -c6
#SBATCH --gres=gpu:pascal:1
#SBATCH -p gpu
module load anaconda3 cuda/10.1 cudnn/10-74
source activate mmmot
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python -u main-ss.py
