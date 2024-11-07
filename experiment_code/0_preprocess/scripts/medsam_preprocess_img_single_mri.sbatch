#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem-per-cpu 16G
#SBATCH --job-name medsam_preprocess_img
#SBATCH --time 0-04:00:00

cd /gpfs/home/kn2347/HCP_MedSAM_project
module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

mri_path=$1
dest_img_encoding_dir=$2
dest_seg_dir=$3
dest_img_dir=$4
id_index_in_path=$5
image_norm_max=$6
python experiment_code/0_preprocess/medsam_preprocess_img.py \
    --start_idx 0 --end_idx 1 \
    --mri_file_pattern ${mri_path} \
    --MedSAM_checkpoint_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/medsam_vit_b.pth \
    --dest_image_encoding_dir ${dest_img_encoding_dir} \
    --dest_img_dir ${dest_img_dir} \
    --id_index_in_path ${id_index_in_path} \
    --image_norm_max ${image_norm_max} \
    --device cuda
