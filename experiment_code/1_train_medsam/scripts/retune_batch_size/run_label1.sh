label_arr=(1)
bs_arr=(1 2 4 8 16 32 64 128 256 512)
for i in ${label_arr[@]}
do
    for j in ${bs_arr[@]}
    do
        jobid1=$(sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/1_train_medsam/scripts/retune_batch_size/train_singletask_models_no_yolo_dice_loss_variable_batch_size.sbatch ${i} ${j} | awk '{print $4}')


        sbatch --time=02:00:00 --dependency=afterany:$jobid1 /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/retune_batch_experiments/evaluate_singletask_medsam_dice_variable_batch.sbatch ${i} ${j}
    done
done
