#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=n-5nodes
#SBATCH --mem=400GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short
#SBATCH --time=1-00:00:00
#SBATCH --exclude=a100-4020


set -x -e

module purge
module add anaconda3/gpu/2022.10
module load anaconda3/gpu/2022.10
module load gcc/11.2.0
conda activate yolov10
module purge
module load slurm/current

export pct=$1
export tag=$2
to_dir=/gpfs/data/luilab/karthik/pediatric_seg_proj/yolov10_results/${pct}
mkdir -p ${to_dir}



cd /gpfs/home/kn2347/yolov10
yolo detect predict model=/gpfs/home/kn2347/yolov10/runs/detect/yolov10_hcp_${pct}/weights/best.pt source=/gpfs/data/cbi/hcp/hcp_ya/hcp_ya_slices_npy/dir_structure_for_yolov7/${tag}/images conf=0.01 iou=0 imgsz=256 project=${to_dir} name=${tag}_run device=0 save_conf=True save_txt=True save=False
