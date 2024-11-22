label_arr=(1 5 7)
batchsz_arr=(8 4 16)
frac_arr=(1 2 3 4 5 7 10 15 20 50 100 250 500 891)
log_arr=(5 5 5 5 5 5 1 1 1 1 1 1 1 1)
lr=0.0005 # learning rate

for i in "${!label_arr[@]}"
do
    idx=0
    batch_size=${batchsz_arr[$i]}
    for j in "${!frac_arr[@]}"
    do
        echo ${label_arr[$i]} ${frac_arr[$j]} ${log_arr[$j]} ${batch_size}
        workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_analysis_11-12-24/MedSAM
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/1_train_medsam/scripts/subset_samples_experiments/with_repeats_11-11-24/train_and_validate_subset_singletask_medsam_labels_dice_with_repeat.sbatch ${label_arr[$i]} ${frac_arr[$j]} ${log_arr[$j]} ${lr} ${workdir} ${batch_size}
        let idx=${idx}+1
    done
done
