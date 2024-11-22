#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=1
#SBATCH --mem=25G
#SBATCH --partition gpu8_medium
#SBATCH --job-name singletask
#SBATCH --time 1-00:00:00
#SBATCH --array=1-102
###SBATCH --exclude=a100-4020,a100-4018
cd /gpfs/home/kn2347/HCP_MedSAM_project
set -x -e

# Training setup
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
export NNODES=$SLURM_NNODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs

module purge
module add anaconda3/gpu/2022.10
module load anaconda3/gpu/2022.10
module load gcc/11.2.0
conda activate medsam
module purge
module load slurm/current

export CUDA_VISIBLE_DEVICES='0'
export workdir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/medsam_yolo_retrain_dice_11-18-24'
export label=$SLURM_ARRAY_TASK_ID
export lr=0.0005

mkdir -p ${workdir}/val/${label}/${lr}/at_least_100_pixels

if ls ${workdir}/val/${label}/${lr}/at_least_100_pixels/*.pkl 1> /dev/null 2>&1; then 
    exit 1
fi

for (( i=0; i < $SLURM_NTASKS; ++i ))
do
    srun -lN1 --mem=25G -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python experiment_code/4_evaluate/full_eval_val_test.py \
    --model_type singletask_yolov7_prompted \
    --explicit_model_path ${workdir}/training/${label}/${lr}/*/medsam_model_best.pth \
    --explicit_yolo_bbox_dataframe_path /gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/yolov10/100/${label}/path_df_only_with_bbox_yolov10.csv \
    --explicit_dataframe_to_merge_with_yolo_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv \
    --dataframe_mask_npy_path /gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_eval_dfs/at_least_100_mask_arrays/label${label}.npy \
    --label ${label} \
    --tag val \
    -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
    --batch_size 64 \
    --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv \
    --output_dir ${workdir}/val/${label}/${lr}/"at_least_100_pixels >> ${workdir}/val/${label}/${lr}/at_least_100_pixels/log_for_${SLURM_JOB_ID}.log 2>&1 &
done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"