#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=RLQwen-%j.out
#SBATCH --mail-user=wbbcor001@myuct.ac.za
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR"
module load python/miniconda3-py3.12
eval "$(conda shell.bash hook)"
conda activate qwen-lora
source grpo-env/bin/activate

srun accelerate launch --config_file deepspeed_zero3.yaml finetune_and_grpo.py
