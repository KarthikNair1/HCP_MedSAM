label_arr=(1 5 18 58)
lr_arr=(0.00005 0.0001 0.0005 0.001)
for i in ${label_arr[@]}
do
    for j in ${lr_arr[@]}
    do
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/2_train_unet/scripts/retune_lr_experiments/train_singletask_unet_variable_lr.sh ${i} ${j}
    done
done
