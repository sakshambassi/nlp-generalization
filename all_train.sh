for lang in 'fr' 'es' 'de' 'tr' 'el' 'bg' 'ru' 'ar' 'vi' 'th' 'zh' 'hi' 'sw' 'ur'; do
    sbatch ./slurms/greene_run.slurm $lang
done