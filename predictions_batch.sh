#!/bin/bash
#SBATCH --job-name=qwen-lora-finetune
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=qwen-lora-%j.out
#SBATCH --mail-user=wbbcor001@myuct.ac.za
#SBATCH --mail-type=ALL

module load python/miniconda3-py3.12
eval "$(conda shell.bash hook)"
conda activate qwen-lora

cd "$SLURM_SUBMIT_DIR"

echo "Python path: $(which python)"
echo "Python version:"
python --version
echo "Transformers version:"
python -c "import transformers; print(transformers.__version__)"
echo "Environment packages:"
pip list | grep transformers


cd "$SLURM_SUBMIT_DIR"
srun python generate_cot_predictions.py
