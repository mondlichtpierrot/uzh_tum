#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-23:59:59
#SBATCH --output=job.out
#SBATCH sbatch slurm_submit.sh

srun python train_reconstruct.py --root1 /net/cephfs/home/pebel/scratch/SEN12MSCRTS --root2 /net/cephfs/home/pebel/scratch/SEN12MSCRTS_val_test