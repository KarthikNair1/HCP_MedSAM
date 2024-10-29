model_type="singletask_unprompted"
mri_id="151425"
explicit_dataframe_path="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/skull_stripped_hcp_experiment_10-24-24/skull_stripped_df_medsam.csv"
output_dir="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/skull_stripped_hcp_experiment_10-24-24/segmentations"
sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/scripts/generate_segmentation_for_mri.sbatch ${model_type} ${mri_id} ${explicit_dataframe_path} ${output_dir}