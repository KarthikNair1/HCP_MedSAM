#label_arr=(1 5 7)
label_arr=(1 5 7)
frac_arr=(1 2 3 4 5 7 10 15 20 50 100 250 500 891)
lr=0.0005 # learning rate

for i in "${!label_arr[@]}"
do
    for j in "${!frac_arr[@]}"
    do
        workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_analysis_11-12-24/MedSAM
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/subsets_sample_experiments/with_repeats/validate_subset_singletask_medsam_labels_dice_with_repeat_at_least_100.sbatch ${label_arr[$i]} ${frac_arr[$j]} ${lr} ${workdir}

        #workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_analysis_11-12-24/UNet
        #sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/subsets_sample_experiments/with_repeats/validate_subset_singletask_unet_labels_dice_with_repeat_at_least_100.sbatch ${label_arr[$i]} ${frac_arr[$j]} ${lr} ${workdir}
    done
done
