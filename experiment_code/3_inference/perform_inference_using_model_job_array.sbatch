#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem-per-cpu 16G
#SBATCH --partition a100_short
#SBATCH --job-name medsam_perform_inference
#SBATCH --time 1-00:00:00

module purge
module add anaconda3/gpu/2022.10
conda activate medsam
module purge

python /gpfs/home/kn2347/test/perform_inference_using_model_multinode.py --node_rank 0