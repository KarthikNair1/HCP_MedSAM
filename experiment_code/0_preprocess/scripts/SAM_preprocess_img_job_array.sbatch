#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu8_medium
#SBATCH --mem-per-cpu 16G
#SBATCH --job-name sam_preprocess_img
#SBATCH --array=0-1113:10
#SBATCH --time 0-04:00:00


cd /gpfs/home/kn2347/HCP_MedSAM_project

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

python /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/0_preprocess/medsam_preprocess_img.py \
    --start_idx ${SLURM_ARRAY_TASK_ID} --end_idx $((SLURM_ARRAY_TASK_ID+10)) \
    --mri_file_pattern "/gpfs/data/cbi/hcp/hcp_seg/data_orig/*/mri/T1.mgz" \
    --MedSAM_checkpoint_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/segment_anything/sam_vit_b_01ec64.pth \
    --dest_image_encoding_dir /gpfs/data/cbi/hcp/hcp_ya/hcp_ya_slices_npy/SAM_image_encoded_slices \
    --device cuda
