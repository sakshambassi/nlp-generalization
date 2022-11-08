#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64GB
#SBATCH --time=28:00:00
#SBATCH --gres=gpu:1

echo "Training scheduled for language : $1 "
eval "$(conda shell.bash hook)"
conda activate nlp_ra
python $PWD/xnli_bert_baseline.py --language $1
