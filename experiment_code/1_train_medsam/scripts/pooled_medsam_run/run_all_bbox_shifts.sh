bbox_shift=(0 3 5 10 15)
for shift in "${bbox_shift[@]}"
do
    sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/1_train_medsam/scripts/pooled_medsam_run/train_singletask_models_oracle_lr5e-4_all_labels_dice_vary_bbox_shift.sbatch $shift
done

