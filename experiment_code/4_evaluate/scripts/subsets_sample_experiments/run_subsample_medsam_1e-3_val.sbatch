#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=100GB
#SBATCH --partition a100_short
#SBATCH --job-name singletask_medsam_val_lr5e-4
#SBATCH --time 1-00:00:00
#SBATCH --array=1
#SBATCH --exclude=a100-4023,a100-4024,a100-4018,a100-4019,a100-4020
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

export label=$1
export pct=$2

export outdir="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subsets_expt_medsam_9-17-24"
mkdir -p ${outdir}/val/0.001/${label}/${pct}

if ls ${outdir}/val/0.001/${label}/${pct}/*.pkl 1> /dev/null 2>&1; then 
    exit 1
fi

for (( i=0; i < $SLURM_NTASKS; ++i ))
do
    srun -lN1 --mem=100G -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python experiment_code/4_evaluate/full_eval_val_test.py \
    --model_type singletask_unprompted \
    --explicit_model_path /gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subsets_expt_medsam_9-17-24/training/0.001/${label}/${pct}/*/*model_best.pth \
    --explicit_dataset_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv \
    --label ${label} \
    --tag val \
    -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
    --batch_size 64 \
    --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv \
    --output_dir ${outdir}/val/0.001/${label}/${pct}" >> ${outdir}/val/0.001/${label}/${pct}/log_for_${SLURM_JOB_ID}.log 2>&1 &

done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"