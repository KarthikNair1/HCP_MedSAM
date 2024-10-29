mri_path='/gpfs/data/luilab/karthik/MICCAI_BraTS2020_TrainingData/images/images001_slice*.png'
seg_path='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/brats_segmentations_10-17-24/medsam/001/singletask_seg_all.npy'
output_dir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/brats_segmentations_10-17-24/region_animations/001/medsam'

sbatch ./experiment_code/5_visualize/scripts/generate_gif_viz_for_mri_stack.sbatch \
    "${mri_path}" ${seg_path} ${output_dir}