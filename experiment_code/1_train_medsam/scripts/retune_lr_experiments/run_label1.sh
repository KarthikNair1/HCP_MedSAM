label_arr=(1)
lr_arr=(0.00005 0.0001 0.0005 0.001 0.0025 0.005 0.01)
for i in ${label_arr[@]}
do
    for j in ${lr_arr[@]}
    do
        jobid1=$(sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/1_train_medsam/scripts/retune_lr_experiments/train_singletask_models_no_yolo_dice_loss_variable_lr.sbatch ${i} ${j} | awk '{print $4}')


        sbatch --time=02:00:00 --dependency=afterany:$jobid1 /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/retune_lr_experiments/evaluate_singletask_medsam_dice_variable_lr.sbatch ${i} ${j}
    done
done
