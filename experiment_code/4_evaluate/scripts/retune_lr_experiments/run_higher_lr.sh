label_arr=(1 5 18 58)
lr_arr=(0.0025 0.005 0.01)
for i in ${label_arr[@]}
do
    for j in ${lr_arr[@]}
    do
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/retune_lr_experiments/evaluate_singletask_medsam_variable_lr.sbatch ${i} ${j}
        #sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/retune_lr_experiments/evaluate_singletask_unet_variable_lr.sbatch ${i} ${j}
    done
done
