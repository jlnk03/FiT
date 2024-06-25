#!/bin/bash
#SBATCH --job-name=fit_train
#SBATCH --output=/storage/user/s0095/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
python train_lightning.py  --model FiT-B/2 --global-batch-size 128  --feature-path /usr/prakt/s0097/latent_two --ckpt-every 10_000 --log-every 50
echo "Job finished"
