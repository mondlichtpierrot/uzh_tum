#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15000
#SBATCH --time=6-23:59:59
#SBATCH --output=job2.out

srun python train_reconstruct.py --root1 /net/cephfs/home/pebel/scratch/SEN12MSCRTS --root2 /net/cephfs/home/pebel/scratch/SEN12MSCRTS_val_test --input_t 3 --region all --loss combined --perceptual None --experiment_name utae_S1S2_t3_L1SSIM_all