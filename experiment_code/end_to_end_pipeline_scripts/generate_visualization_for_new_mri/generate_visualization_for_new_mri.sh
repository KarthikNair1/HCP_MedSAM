mri_id_name=$1
mri_path=$2
output_dir=$3
model_type=$4
id_index_in_path=$5 # this value gives the index in the full filepath to the MRI in which the ID name is found (e.g. -2 means the MRI path takes the form */<ID>/*.nii)
image_norm_max=$6


dest_img_encoding_dir=${output_dir}/inputs/medsam_embeddings
mkdir -p ${dest_img_encoding_dir}
dest_seg_dir=${output_dir}/inputs/seg
mkdir -p ${dest_seg_dir}
dest_img_dir=${output_dir}/inputs/pngs
mkdir -p ${dest_img_dir}

# preprocess MRI


#sbatch --wait /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/0_preprocess/scripts/medsam_preprocess_img_single_mri.sbatch \
#    ${mri_path} ${dest_img_encoding_dir} ${dest_seg_dir} ${dest_img_dir} ${id_index_in_path}
sh /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/0_preprocess/scripts/medsam_preprocess_img_single_mri.sbatch \
    ${mri_path} ${dest_img_encoding_dir} ${dest_seg_dir} ${dest_img_dir} ${id_index_in_path} ${image_norm_max}

# generate dataframe
python /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/0_preprocess/generate_dataframe_for_single_mri.py \
    --model_type ${model_type} --mri_id ${mri_id_name} \
    --img_encoding_dir_pattern "${dest_img_encoding_dir}/${mri_id_name}/*.npy" \
    --seg_dir_pattern "${dest_seg_dir}/${mri_id_name}/seg_*.npy" \
    --img_dir_pattern "${dest_img_dir}/${mri_id_name}/*.png" \
    --output_dir ${output_dir}/inputs/${mri_id_name}

# inference using a model
explicit_dataframe_path=${output_dir}/inputs/${mri_id_name}/path_df_${model_type}.csv
inference_output_dir=${output_dir}/segmentations/${mri_id_name}/${model_type}
mkdir -p ${inference_output_dir}
#sbatch --wait /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/scripts/generate_segmentation_for_mri.sbatch \
#    ${model_type} ${mri_id_name} ${explicit_dataframe_path} ${inference_output_dir}
sh /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/scripts/generate_segmentation_for_mri.sbatch \
    ${model_type} ${mri_id_name} ${explicit_dataframe_path} ${inference_output_dir}

png_pattern="${dest_img_dir}/${mri_id_name}/*.png"
seg_path=${inference_output_dir}/${mri_id_name}/singletask_seg_all.npy

# run gif visualization
gif_viz_dir=${output_dir}/region_animations/${mri_id_name}/${model_type}
#sbatch --wait /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/5_visualize/scripts/generate_gif_viz_for_mri_stack.sbatch \
#    "${png_pattern}" ${seg_path} ${gif_viz_dir}
sh /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/5_visualize/scripts/generate_gif_viz_for_mri_stack.sbatch \
    "${png_pattern}" ${seg_path} ${gif_viz_dir}

