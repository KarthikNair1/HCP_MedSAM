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

python /gpfs/home/kn2347/test/medsam_preprocess_img.py --start_idx ${SLURM_ARRAY_TASK_ID} --end_idx $((SLURM_ARRAY_TASK_ID+10))