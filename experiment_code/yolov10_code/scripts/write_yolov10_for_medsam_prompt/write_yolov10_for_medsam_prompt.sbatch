#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=n-5nodes
#SBATCH --mem=100GB
#SBATCH --partition=cpu_short
#SBATCH --time=12:00:00


set -x -e

module purge
module add anaconda3/gpu/2022.10
module load anaconda3/gpu/2022.10
module load gcc/11.2.0
conda activate medsam
module purge
module load slurm/current

export pct=$1
to_dir=/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/yolov10/${pct}
mkdir -p ${to_dir}



cd /gpfs/home/kn2347/HCP_MedSAM_project

python /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/yolov10_code/write_yolov7_for_medsam_prompt.py \
    --root_dir /gpfs/data/luilab/karthik/pediatric_seg_proj/yolov10_results/${pct} \
    --confidence_threshold 0.01 \
    --full_df_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv \
    --save_dir ${to_dir}