label_arr=(1 5 18 58)
frac_arr=(0 0.001 0.01 0.05 0.1 0.25 1)
for i in ${label_arr[@]}
do
    for j in ${frac_arr[@]}
    do
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/compare_blank_training_experiments/evaluate_singletask_unet_val_lr1e-4_for_blank.sbatch ${i} ${j}
    done
done
