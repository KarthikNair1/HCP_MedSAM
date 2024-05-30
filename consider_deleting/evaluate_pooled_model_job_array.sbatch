#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem-per-cpu 16G
#SBATCH --partition a100_short
#SBATCH --job-name pooled_val
#SBATCH --array=3
#SBATCH --time 1-00:00:00

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

python /gpfs/home/kn2347/test/evaluate/evaluate_pooled_model.py --node_rank ${SLURM_ARRAY_TASK_ID}