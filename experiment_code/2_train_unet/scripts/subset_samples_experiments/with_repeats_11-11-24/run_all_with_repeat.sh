label_arr=(1 5 7)
batchsz_arr_unet=(8 16 4)
frac_arr=(1 2 3 4 5 7 10 15 20 50 100 250 500 891)
log_arr=(5 5 5 5 5 5 1 1 1 1 1 1 1 1)
lr=0.0005 # learning rate

for i in "${!label_arr[@]}"
do
    batch_size=${batchsz_arr_unet[$i]}
    for j in "${!frac_arr[@]}"
    do 
        workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_analysis_11-12-24/UNet
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/2_train_unet/scripts/subset_samples_experiments/with_repeats_11-11-24/train_subset_singletask_unet_dice_with_repeat.sbatch ${label_arr[$i]} ${frac_arr[$j]} ${log_arr[$j]} ${lr} ${workdir} ${batch_size}
    done
done
