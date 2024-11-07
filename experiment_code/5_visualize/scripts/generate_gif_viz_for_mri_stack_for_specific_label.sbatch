#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu8_medium
#SBATCH --mem-per-cpu 16G
#SBATCH --job-name generate_seg
#SBATCH --time 1-00:00:00

cd /gpfs/home/kn2347/HCP_MedSAM_project

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

mri_path=$1
seg_path=$2
output_dir=$3
label=$4

mkdir -p ${output_dir}
if ls ${output_dir}/*.gif 1> /dev/null 2>&1; then 
    exit 1
fi

python /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/5_visualize/generate_gif_viz_for_mri_stack.py \
    --mri_path "${mri_path}" \
    --seg_path ${seg_path} \
    --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv \
    --output_dir ${output_dir} \
    --do_region_gifs \
    --single_label ${label} \
    --force_color