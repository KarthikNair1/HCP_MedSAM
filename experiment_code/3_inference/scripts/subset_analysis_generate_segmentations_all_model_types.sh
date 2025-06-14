label_arr=(1 5 7)
frac_arr=(1 2 3 4 5 7 10 15 20 50 100 250 500 891)
#frac_arr=(1)
repeat_num_arr=(0 1 0)
model_types=("singletask_unprompted" "singletask_unet")
training_model_names=("MedSAM" "UNet")
model_suffix_patterns=("medsam_model_best.pth" "*-best_model.pth")

df_path_medsam=/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv
df_path_unet=/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_unet.csv
df_paths=(${df_path_medsam} ${df_path_unet})

for model_type_idx in "${!model_types[@]}"
do
    model_type=${model_types[$model_type_idx]}
    model_pattern="${model_suffix_patterns[$model_type_idx]}"
    for i in ${label_arr[@]}
    do
        label=$i
        idx=0
        for j in "${!frac_arr[@]}"
        do
            frac=${frac_arr[$j]}
            #repeat_num=${repeat_num_arr[$model_type_idx]} <- this line was what was originally run, but it's wrong
            repeat_num=${repeat_num_arr[$i]} 
            echo ${i} ${model_type} ${frac} ${repeat_num}

            mri_id=162935
            training_name=${training_model_names[$model_type_idx]}
            #explicit_dataframe_path=/gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv
            explicit_dataframe_path=${df_paths[$model_type_idx]}

            workdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_expt_gif_visualizations_12-10-24/segmentations/${model_type}/${i}/${frac}
            if [[ "$model_type" == "singletask_unprompted" ]]; then
                explicit_model_path="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_analysis_11-12-24/${training_name}/training/*/${label}/${frac}/${repeat_num}/*/medsam_model_best.pth"
            elif [[ "$model_type" == "singletask_unet" ]]; then
                explicit_model_path="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_analysis_11-12-24/${training_name}/training/*/${label}/${frac}/${repeat_num}/*-best_model.pth"
            fi
            
            label=${i}
            sbatch /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/scripts/generate_segmentation_for_mri_with_explicit_model_path.sbatch ${model_type} ${mri_id} ${explicit_dataframe_path} ${workdir} ${explicit_model_path} ${label}
            let idx=${idx}+1
        done
    done
done

