#label_arr=(1 71 7)
label_arr=(1)
frac_arr=(1 2 3 4 5 7 10 15 20 50 100 250 500 891)
#repeats=(0 1 0)
#repeats=(0)
for i in ${label_arr[@]}
do
    # let idx = 0
    idx=0
    for j in "${!frac_arr[@]}"
    do
        for repeat_num in 0 1 2
        do
            #repeat_num=${repeats[$idx]}

            model_type="singletask_unet"
            mri_id="162935"

            explicit_model_path="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_analysis_11-12-24/UNet/training/0.0005/${i}/${frac_arr[$j]}/${repeat_num}/*-best_model.pth"
            explicit_model_path_label_index=-4
            explicit_dataframe_path=/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_unet.csv
            explicit_yolo_bbox_dataframe_path="None"
            label=${i}
            output_dir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_expt_gif_visualizations_12-30-24/segmentations/${model_type}/${i}/${frac_arr[$j]}/${repeat_num}
            
            sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/scripts/generate_segmentation_for_mri_with_explicit_model_path.sbatch \
                ${model_type} ${mri_id} "${explicit_model_path}" ${explicit_model_path_label_index} ${explicit_dataframe_path} ${explicit_yolo_bbox_dataframe_path} ${label} ${output_dir}
        done
    done
    # add 1 to idx
    idx=$((idx+1))
done

