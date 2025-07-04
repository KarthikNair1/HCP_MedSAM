model_type="singletask_unet"
mri_id="162935"
explicit_model_path="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/unet_retrain_all_labels_9-9-24/training/*/*-best_model.pth"
explicit_model_path_label_index=-2
explicit_dataframe_path="/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_unet.csv"
explicit_yolo_bbox_dataframe_path="None"
label="None"
output_dir="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/hcp_test_segmentations_12-30-24/segmentations/singletask_unet"

sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/scripts/generate_segmentation_for_mri_with_explicit_model_path.sbatch \
    ${model_type} ${mri_id} "${explicit_model_path}" ${explicit_model_path_label_index} ${explicit_dataframe_path} ${explicit_yolo_bbox_dataframe_path} ${label} ${output_dir}
