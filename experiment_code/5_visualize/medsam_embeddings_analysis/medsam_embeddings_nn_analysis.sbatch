#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 32G
#SBATCH --time 0-08:00:00
#SBATCH --job-name embedding-nn
#SBATCH --output embedding_logs-%J.log

module purge
module add anaconda3/gpu/2022.10
module load anaconda3/gpu/2022.10
module load gcc/11.2.0
conda activate medsam
module purge
module load slurm/current

python /gpfs/home/kn2347/test/medsam_embeddings_nn_analysis.py