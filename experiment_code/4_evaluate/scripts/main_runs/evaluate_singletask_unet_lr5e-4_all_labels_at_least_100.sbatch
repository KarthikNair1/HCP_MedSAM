#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=25GB
#SBATCH --partition a100_short
#SBATCH --job-name singletask_unet_val
#SBATCH --time 1-00:00:00
#SBATCH --array=1-102
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
export workdir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/unet_retrain_all_labels_9-9-24'
export label=$SLURM_ARRAY_TASK_ID
mkdir -p ${workdir}/val/${label}/0.0005/at_least_100
if ls ${workdir}/val/${label}/0.0005/at_least_100/*.pkl 1> /dev/null 2>&1; then 
    exit 1
fi

for (( i=0; i < $SLURM_NTASKS; ++i ))
do
    srun -lN1 --mem=25G -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python experiment_code/4_evaluate/full_eval_val_test.py \
    --model_type singletask_unet \
    --explicit_model_path ${workdir}/training/${label}/*.pth \
    --explicit_dataset_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_unet.csv \
    --dataframe_mask_npy_path /gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_eval_dfs/at_least_100_mask_arrays/label${label}.npy \
    --label ${label} \
    --tag val \
    -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
    --batch_size 64 \
    --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv \
    --output_dir ${workdir}/val/${label}/0.0005/at_least_100" >> ${workdir}/val/${label}/0.0005/at_least_100/log_for_${SLURM_JOB_ID}.log 2>&1 &

done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"