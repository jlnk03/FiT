#!/bin/bash
#SBATCH --job-name=masked_flexible_diffusion_train
#SBATCH --output=/usr/prakt/s0095/GSU/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --nodelist=node[3,8,9,10,11,12,13,14,15,16,17,18]
#SBATCH --time=7-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
srun python train_lightning_diffusers_final.py  --model FiT-B/2 --global-batch-size 128 --epochs 100 --feature-path /storage/slurm/schnaus/gsu_2024/latentnew_train --feature-val-path /storage/slurm/schnaus/gsu_2024/latentnew_val --results-dir /storage/slurm/schnaus/gsu_2024/results/masked --ckpt-every 10_000 --log-every 10
echo "Job finished"