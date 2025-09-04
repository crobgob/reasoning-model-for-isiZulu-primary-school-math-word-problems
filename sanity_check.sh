#!/bin/bash
#SBATCH --job-name=san_check
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=san-%j.out
#SBATCH --mail-user=wbbcor001@myuct.ac.za
#SBATCH --mail-type=ALL

module load python/miniconda3-py3.12
eval "$(conda shell.bash hook)"
conda activate qwen-lora

cd "$SLURM_SUBMIT_DIR"
srun sanity_check.py
