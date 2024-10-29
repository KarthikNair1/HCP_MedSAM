#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu 16G
#SBATCH --partition=gpu8_medium
#SBATCH --job-name brats_medsam_preprocess_img
#SBATCH --array=0-49:10
#SBATCH --time 0-04:00:00

cd /gpfs/home/kn2347/HCP_MedSAM_project

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

python /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/other_misc_experiments/brats/medsam_preprocess_img_brats.py \
    --start_idx ${SLURM_ARRAY_TASK_ID} --end_idx $((SLURM_ARRAY_TASK_ID+10)) \
    --mri_file_pattern "/gpfs/data/luilab/karthik/MICCAI_BraTS2020_TrainingData/BraTS20_Training_*/BraTS20_Training_*_t1.nii.gz" \
    --MedSAM_checkpoint_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/medsam_vit_b.pth \
    --dest_image_encoding_dir /gpfs/data/luilab/karthik/MICCAI_BraTS2020_TrainingData/encoded_slices \
    --dest_seg_dir /gpfs/data/luilab/karthik/MICCAI_BraTS2020_TrainingData/segmentations \
    --dest_img_dir /gpfs/data/luilab/karthik/MICCAI_BraTS2020_TrainingData/images \
    --device cuda