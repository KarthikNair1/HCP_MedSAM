#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
###SBATCH --gpus-per-node=a100:1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu 16G
#SBATCH --partition gpu8_medium
#SBATCH --job-name generate_seg
#SBATCH --time 1-00:00:00
#SBATCH --exclude=a100-4004


cd /gpfs/home/kn2347/HCP_MedSAM_project

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

model_type=$1
mri_id=$2
explicit_model_path=$3
explicit_model_path_label_index=$4
explicit_dataframe_path=$5
explicit_yolo_bbox_dataframe_path=$6
label=$7
workdir=$8
foundation_model_type=$9

mkdir -p ${workdir}
if ls ${workdir}/${mri_id}/*.npy 1> /dev/null 2>&1; then 
    exit 1
fi

python /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/3_inference/generate_segmentation_for_mri.py \
    --model_type ${model_type} \
    --mri_id ${mri_id} \
    --explicit_model_path "${explicit_model_path}" \
    --explicit_model_path_label_index ${explicit_model_path_label_index} \
    --foundation_model_type ${foundation_model_type} \
    --explicit_dataframe_path ${explicit_dataframe_path} \
    --explicit_yolo_bbox_dataframe_path ${explicit_yolo_bbox_dataframe_path} \
    --single_label ${label} \
    --batch_size 16 \
    --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv \
    --output_dir ${workdir}


