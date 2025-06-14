#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=make_gt_bbox_df
#SBATCH --mem=16GB
#SBATCH --partition=cpu_dev
#SBATCH --time=04:00:00
#SBATCH --array=1-102

cd /gpfs/home/kn2347/HCP_MedSAM_project
module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

label=${SLURM_ARRAY_TASK_ID}
outdir=/gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/${label}/0.1/isolated_path_df_bboxes_from_ground_truth.csv
if ls ${outdir} 1> /dev/null 2>&1; then 
    exit 1
fi

python /gpfs/home/kn2347/HCP_MedSAM_project/experiment_code/0_preprocess/create_medsam_with_groundtruth_bboxes.py --label ${label}

echo "END TIME: $(date)"