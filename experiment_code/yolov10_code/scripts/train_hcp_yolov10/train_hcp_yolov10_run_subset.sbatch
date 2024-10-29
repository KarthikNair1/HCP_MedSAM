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
export epochs=$2

cd /gpfs/home/kn2347/yolov10
yolo train model=yolov8n.pt data=/gpfs/home/kn2347/yolov10/data/hcp_subset_${pct}.yaml epochs=${epochs} imgsz=256 batch=32 workers=8 name=yolov10_hcp_${pct} device=0