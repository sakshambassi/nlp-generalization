for lang in 'fr' 'es' 'de' 'tr' 'el' 'bg' 'ru' 'ar' 'vi' 'th' 'zh' 'hi' 'sw' 'ur'; do
    output="xnli_fim_${lang}.out"
    err_output="xnli_fim_${lang}.err"
    job="fim_${lang}"
    sbatch --error=$err_output --output=$output --job-name=$job ./slurms/greene_run.slurm $lang
done
