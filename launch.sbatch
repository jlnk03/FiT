#!/bin/bash
#SBATCH --job-name=extract_latents
#SBATCH --output=/storage/user/Duj4vaa+/slurm/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --part=PRACT
#SBATCH --qos=practical_course

echo "Starting the job"
srun python my_script.py
echo "Job finished"
