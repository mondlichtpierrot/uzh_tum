#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=6-23:59:59
#SBATCH --output=job.out

srun python train_reconstruct.py --root1 /net/cephfs/home/pebel/scratch/SEN12MSCRTS --root2 /net/cephfs/home/pebel/scratch/SEN12MSCRTS_val_test --input_t 5 --region all --loss combined --perceptual None --experiment_name utae_S1S2_t5_L1SSIM_all