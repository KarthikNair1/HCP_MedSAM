label_arr=(1 5 7)
frac_arr=(0.1 0.5 1 2.5 5 10 25 50 75 100)


for i in ${label_arr[@]}
do
    idx=0
    for j in "${!frac_arr[@]}"
    do
        echo ${i} ${frac_arr[$j]} ${patience_arr[$j]} ${log_arr[$j]}


        model_type=singletask_unprompted
        mri_id=162935
        explicit_dataframe_path=/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv
        workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_expt_gif_visualizations_12-10-24/segmentations/${model_type}/${i}/${frac_arr[$j]}
        explicit_model_path=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subsets_expt_medsam_9-17-24/training/0.001/${i}/${frac_arr[$j]}/*/medsam_model_best.pth
        label=${i}
        sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/scripts/generate_segmentation_for_mri_with_explicit_model_path.sbatch ${model_type} ${mri_id} ${explicit_dataframe_path} ${workdir} ${explicit_model_path} ${label}
        let idx=${idx}+1
    done
done

