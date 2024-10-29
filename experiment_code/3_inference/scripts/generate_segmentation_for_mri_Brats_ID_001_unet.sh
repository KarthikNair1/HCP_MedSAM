model_type="singletask_unet"
mri_id="001"
explicit_dataframe_path="/gpfs/data/luilab/karthik/pediatric_seg_proj/other_path_dfs/path_df_brats_unet.csv"
output_dir="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/brats_segmentations_9-17-24/unet"

sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/scripts/generate_segmentation_for_mri.sbatch ${model_type} ${mri_id} ${explicit_dataframe_path} ${output_dir}