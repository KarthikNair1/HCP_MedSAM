#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem-per-cpu 16G
#SBATCH --job-name medsam_preprocess_img
#SBATCH --array=0-1113:10
#SBATCH --time 0-04:00:00

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge
cd /gpfs/home/kn2347/HCP_MedSAM_project

python experiment_code/0_preprocess/medsam_preprocess_img.py \
    --start_idx ${SLURM_ARRAY_TASK_ID} --end_idx $((SLURM_ARRAY_TASK_ID+10)) \
    --mri_file_pattern "/gpfs/data/cbi/hcp/hcp_seg/data_orig/*/mri/T1.mgz" \
    --MedSAM_checkpoint_path /gpfs/home/kn2347/MedSAM/medsam_vit_b.pth \
    --dest_image_encoding_dir /gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/pretrained_image_encoded_slices \
    --dest_seg_dir /gpfs/data/luilab/karthik/pediatric_seg_proj/hcp_ya_slices_npy/segmentation_slices \
    --device cuda
