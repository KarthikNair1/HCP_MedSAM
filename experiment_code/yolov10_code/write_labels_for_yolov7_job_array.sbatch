#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --job-name medsam_preprocess_img
#SBATCH --array=0-5
#SBATCH --time 0-04:00:00

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

python /gpfs/home/kn2347/test/write_labels_for_yolov7.py --node_idx $((SLURM_ARRAY_TASK_ID))