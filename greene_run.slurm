#!/bin/bash
#SBATCH --job-name=text_classification
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --output=text_classification.out
#SBATCH --error=error_output.err
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate nlp_ra
python text_classification.py