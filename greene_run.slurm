#!/bin/bash
#SBATCH --job-name=text_classification_RA
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --output=text_classification_RA.out
#SBATCH --error=text_classification_RA.err
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate nlp_ra
python text_classification.py