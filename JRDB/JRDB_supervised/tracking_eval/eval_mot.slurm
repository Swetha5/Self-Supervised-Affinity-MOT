#!/bin/bash
#SBATCH -c6
#SBATCH --gres=gpu:pascal:1
#SBATCH --qos preempt
#SBATCH -p gpu
module load anaconda3 cuda/10 cudnn/10-74
source activate mmmot
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python -u eval_mot.py '/home/c3-0/datasets/JRDB/ground_truth_val_labels/sequences' '/home/aa809504/JRDB-SAMOT/weights/train_with_kitti_weights/max_sample_10/epoch_23'