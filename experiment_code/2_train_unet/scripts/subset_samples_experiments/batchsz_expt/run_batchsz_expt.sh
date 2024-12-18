label_arr=(71)
frac_arr=(0.1 100)
patience_arr=(3 3)
log_arr=(5 1)
batchsz_arr=(1 2 4 8 16 32 64 128 256)
lr=(0.0005)
for u in ${lr[@]}
do
    for i in ${label_arr[@]}
    do
        for j in "${!frac_arr[@]}"
        do
            for batchsz in "${!batchsz_arr[@]}"
            do
                echo ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]} ${batchsz_arr[$batchsz]}
                workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/unet_retrain_dice_batchsz_expt_11-11-24/training
                df_path=/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/${i}/0.1/isolated_path_df_unet.csv
                jobid1=$(sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/2_train_unet/scripts/subset_samples_experiments/batchsz_expt/train_subset_singletask_unet_labels_variable_lr_and_batchsz.sbatch ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]} ${u} ${df_path} ${workdir} ${batchsz_arr[$batchsz]} | awk '{print $4}')

                workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/unet_retrain_dice_batchsz_expt_11-11-24
                sbatch --time=02:00:00 --dependency=afterany:$jobid1 /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/batch_expt/run_subsample_unet_5e-4_batch_expt.sbatch ${i} ${frac_arr[$j]} ${batchsz_arr[$batchsz]} ${workdir}
            done
        done
    done
done


