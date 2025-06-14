mri_path='/gpfs/data/cbi/hcp/hcp_ya/hcp_ya_slices_npy/dir_structure_for_yolov7/test/images/162935_slice*.png'
seg_path='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/hcp_test_segmentations_12-30-24/segmentations/singletask_unprompted/162935/singletask_seg_all.npy'
output_dir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/hcp_test_segmentations_12-30-24/region_animations/singletask_unprompted/162935'

sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/5_visualize/scripts/generate_gif_viz_for_mri_stack.sbatch "${mri_path}" ${seg_path} ${output_dir}