#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem-per-cpu 16G
#SBATCH --partition a100_short
#SBATCH --job-name generate_seg
#SBATCH --time 1-00:00:00

cd /gpfs/home/kn2347/HCP_MedSAM_project

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

model_type=$1
mri_id=$2
explicit_dataframe_path=$3
workdir=$4


python /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/generate_segmentation_for_mri.py \
    --model_type ${model_type} \
    --mri_id ${mri_id} \
    --explicit_dataframe_path ${explicit_dataframe_path} \
    --batch_size 16 \
    --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv \
    --output_dir ${workdir}


