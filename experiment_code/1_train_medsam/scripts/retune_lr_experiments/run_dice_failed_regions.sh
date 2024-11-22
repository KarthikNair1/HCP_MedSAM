label_arr=(96 23 40 72 10 44 33 18)
lr_arr=(0.00001 0.00005 0.0001 0.0005 0.001)
for i in ${label_arr[@]}
do
    for j in ${lr_arr[@]}
    do
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/1_train_medsam/scripts/retune_lr_experiments/train_singletask_models_no_yolo_dice_loss_variable_lr.sbatch ${i} ${j}
    done
done
