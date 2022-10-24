#!/bin/bash
#SBATCH --job-name="xnli_fim_${lang}"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --output="xnli_fim_${lang}.out"
#SBATCH --error=xnli_fim_${lang}.err
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

echo "Training scheduled for language : ${lang}"
eval "$(conda shell.bash hook)"
conda activate nlp_ra
python $PWD/experiments/xnli_bert_fim.py \
 --language $lang
