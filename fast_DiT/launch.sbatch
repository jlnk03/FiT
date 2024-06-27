#!/bin/bash
#SBATCH --job-name=fit_train_dist
#SBATCH --output=/usr/prakt/s0095/GSU/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=48G
#SBATCH --nodelist=node[3,8,9,10,11,12,13,14,15,16,17,18]
#SBATCH --time=1-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
srun python train_lightning.py  --model FiT-B/2 --global-batch-size 128  --feature-path /usr/prakt/s0097/latent_two --feature-val-path /usr/prakt/s0097/latent_val --ckpt-every 10_000 --log-every 10
echo "Job finished"