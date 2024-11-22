label_arr=(1 5 7)
frac_arr=(0.1 100)
patience_arr=(3 10)
log_arr=(5 1)
#lr=(0.0005 0.001 0.005 0.01)
lr=(0.0005)
for u in ${lr[@]}
do
    for i in ${label_arr[@]}
    do
        idx=0
        for j in "${!frac_arr[@]}"
        do
            echo ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]}
            workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/medsam_retrain_dice_longtrain_11-8-24/training
            sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/1_train_medsam/scripts/subset_samples_experiments/new2/longtrain_expt/train_subset_singletask_medsam_labels_dice_without_earlystopping.sbatch ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]} ${u} ${workdir}
            let idx=${idx}+1
        done
    done
done
