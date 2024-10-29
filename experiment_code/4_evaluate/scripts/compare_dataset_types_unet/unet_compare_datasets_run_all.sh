export model_with_blanks='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/unet_singletask_testing_9-3-24_retraining/lr1e-4/singletask_unet-label5-20240903-034705-best_model.pth'
export model_without_blanks='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/unet_singletask_testing_5-26-24/logs_training/lr1e-4/singletask_unet-label5-20240722-172502-best_model.pth'
export data_with_blanks='/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_unet.csv'
export data_without_blanks='/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df/baseline_unet/all_labels_df.csv'

# model without blank on data without blank
#sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/compare_dataset_types_unet/evaluate_singletask_unet_val_lr1e-4_label5_for_comparison_datasets.sbatch ${model_without_blanks} ${data_without_blanks} 'm-_d-'

# model without blank on data WITH blank
#sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/compare_dataset_types_unet/evaluate_singletask_unet_val_lr1e-4_label5_for_comparison_datasets.sbatch ${model_without_blanks} ${data_with_blanks} 'm-_d+'

# model WITH blank on data without blank
sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/compare_dataset_types_unet/evaluate_singletask_unet_val_lr1e-4_label5_for_comparison_datasets.sbatch ${model_with_blanks} ${data_without_blanks} 'm+_d-'

# model WITH blank on data WITH blank
#sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/4_evaluate/scripts/compare_dataset_types_unet/evaluate_singletask_unet_val_lr1e-4_label5_for_comparison_datasets.sbatch ${model_with_blanks} ${data_with_blanks} 'm+_d+'

