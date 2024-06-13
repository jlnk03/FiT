#!/bin/bash
#SBATCH --job-name=extract_latents
#SBATCH --output=/storage/user/s0095/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --part=PRACT

echo "Starting the job"
python preprocess/preprocess.py
echo "Job finished"
