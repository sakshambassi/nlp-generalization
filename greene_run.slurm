#!/bin/bash
#SBATCH --job-name=xnli_fim
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --output=xnli_fim.out
#SBATCH --error=xnli_fim.err
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2

eval "$(conda shell.bash hook)"
conda activate nlp_ra
python xnli_bert_fim.py
