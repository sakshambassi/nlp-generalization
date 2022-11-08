output="xnli_baseline_en.out"
err_output="xnli_baseline_en.err"
job="baseline_en"
sbatch --error=$err_output --output=$output --job-name=$job ./slurms/greene_run.slurm 'en'

