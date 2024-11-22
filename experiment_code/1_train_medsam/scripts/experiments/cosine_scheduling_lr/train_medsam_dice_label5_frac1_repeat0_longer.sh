export label=5
export frac=1
export lr=0.0005
export repeat=0
export workdir="/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/hyperparam_tuning/learning_scheduling_how_long_subset"
log_val_every=1000
WORLD_SIZE=1
batch_size=4

mkdir -p ${workdir}/training/${lr}/${label}/${frac}/${repeat}

export CUDA_VISIBLE_DEVICES=0

python experiment_code/1_train_medsam/train_multi_gpus_modified_multiclass.py \
        --data_frame_path /gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/${label}/0.1/isolated_path_df_constant_bbox.csv \
        --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
        --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/class_mappings/label${label}_only_name_class_mapping.csv \
        -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/subset_train_id_dfs_pooled/repeat_experiments/${frac}/repeat${repeat}.pkl \
        -task_name singletask_medsam_no_yolo \
        --wandb_run_name label5_subset_LONGERTRAIN \
        -checkpoint /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/medsam_vit_b.pth \
        -work_dir ${workdir}/training/${lr}/${label}/${frac}/${repeat} \
        -label_id 1 \
        -batch_size ${batch_size} \
        -val_batch_size 256 \
        -num_workers 1 \
        -num_epochs 50000 \
        --lambda_dice 1 \
        -lr ${lr} \
        --log_val_every ${log_val_every} \
        -use_wandb True \
        -use_amp \
        --suppress_train_debug_imgs \
        --world_size ${WORLD_SIZE} \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank 0 