#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

echo "Training scheduled for language : ${lang}"
eval "$(conda shell.bash hook)"
conda activate nlp_ra
python $PWD/experiments/xnli_bert_fim.py \
 --language $lang
