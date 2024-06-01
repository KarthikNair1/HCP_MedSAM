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

python /gpfs/home/kn2347/test/perform_inference_using_model_multinode.py \
    --model_path /gpfs/home/kn2347/results/models_8-9-23/scratch_loss_reweighted_lr1e-4_ce_only_longer3_model_20230804-133537/model_best.pth \
    --data_frame_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv \
    --train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
    --df_starting_mapping_path /gpfs/home/kn2347/MedSAM/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/MedSAM/darts_name_class_mapping_processed.csv \
    --output_dir /gpfs/data/luilab/karthik/pediatric_seg_proj/saved_round1_segmentations_bbox \
    --num_nodes 1 \
    --node_rank 0