label_arr=(1 5 18 58)
frac_arr=(0.1 0.5 1 2.5 5 10 25 50 75 100)


for i in ${label_arr[@]}
do
    idx=0
    for j in "${!frac_arr[@]}"
    do

        mri_path='/gpfs/data/cbi/hcp/hcp_ya/hcp_ya_slices_npy/dir_structure_for_yolov7/val/images/151425_slice*.png'
        seg_path=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_expt_gif_visualizations_10-30-24/segmentations/singletask_unprompted/${i}/${frac_arr[$j]}/151425/singletask_seg_all.npy
        output_dir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_expt_gif_visualizations_10-30-24/region_animations/151425/singletask_unprompted/${i}/${frac_arr[$j]}
        label=${i}

        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/5_visualize/scripts/generate_gif_viz_for_mri_stack_for_specific_label.sbatch "${mri_path}" ${seg_path} ${output_dir} ${label}
        let idx=${idx}+1
    done
done

