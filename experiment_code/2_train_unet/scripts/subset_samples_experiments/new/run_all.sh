label_arr=(1 5 18 58)
frac_arr=(0.1 0.5 1 2.5 5 10 25 50 75 100)
patience_arr=(3 5 10 10 10 10 10 10 10 10)
log_arr=(5 2 1 1 1 1 1 1 1 1)
lr=(0.0005)
for u in ${lr[@]}
do
    for i in ${label_arr[@]}
    do
        idx=0
        for j in "${!frac_arr[@]}"
        do
            echo ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]}
            df_path=/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/${i}/0.1/isolated_path_df_unet.csv
            workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subsets_expt_unet_9-12-24/training
            sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/2_train_unet/scripts/subset_samples_experiments/new/train_subset_singletask_unet_labels_variable_lr.sbatch ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]} ${u} ${df_path} ${workdir}
            let idx=${idx}+1
        done
    done
done
