#!/bin/bash
#SBATCH --job-name=fit_FID_50k
#SBATCH --output=/usr/prakt/s0095/GSU/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --nodelist=node[3,8,9,10,11,12,13,14,15,16,17,18]
#SBATCH --time=2-00:00:00

echo "Starting the job"
echo "Job ID: $SLURM_JOB_ID"
srun python sample_lightning.py --checkpoint_path /storage/slurm/schnaus/gsu_2024/results/checkpoints/epoch=99-step=500500.ckpt --num_samples 50_000 --cfg_scale 1.5
echo "Job finished"