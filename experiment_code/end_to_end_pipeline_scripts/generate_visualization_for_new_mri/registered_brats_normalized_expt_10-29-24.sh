mri_id_name='brats_001_registered'
mri_path='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/registered_brats_normalized_expt_10-29-24/inputs/brats_001_registered/registered_t1.nii'
output_dir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/registered_brats_normalized_expt_10-29-24'
model_type='singletask_unprompted'
id_index_in_path=-2
image_norm_max=200

sh /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/end_to_end_pipeline_scripts/generate_visualization_for_new_mri/generate_visualization_for_new_mri.sh \
    ${mri_id_name} ${mri_path} ${output_dir} ${model_type} ${id_index_in_path} ${image_norm_max}