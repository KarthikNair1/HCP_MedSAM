#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 32G
#SBATCH --time 1-00:00:00
#SBATCH --job-name write-yolov7-for-medsam-prompt

module add anaconda3/gpu/2022.10
module load anaconda3/gpu/2022.10
module load gcc/11.2.0
conda activate medsam
module purge

python /gpfs/home/kn2347/test/yolov7/write_yolov7_for_medsam_prompt.py
