for lang in 'fr' 'es' 'de' 'tr' 'el' 'bg' 'ru' 'ar' 'vi' 'th' 'zh' 'hi' 'sw' 'ur'; do
    output="xnli_base_${lang}.out"
    err_output="xnli_base_${lang}.err"
    job="base_${lang}"
    sbatch --error=$err_output --output=$output --job-name=$job ./slurms/greene_run.slurm $lang
done
