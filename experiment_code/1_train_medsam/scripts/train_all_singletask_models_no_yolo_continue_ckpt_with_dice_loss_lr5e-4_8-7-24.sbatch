#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=singletask_medsam
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short

#SBATCH --output=logs/medsam_mgpus_%x-%j.out
#SBATCH --error=logs/medsam_mgpus_%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --array=1-102
#SBATCH --exclude=a100-4020,a100-4023,a100-4017,a100-4018,a100-4019,a100-4011

cd /gpfs/home/kn2347/HCP_MedSAM_project
set -x -e

# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR

# Training setup
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

## Master node setup
MAIN_HOST=`hostname -s`
export MASTER_ADDR=$MAIN_HOST

# Get a free port using python
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

export NNODES=$SLURM_NNODES
#NODE_RANK=$SLURM_PROCID ## do i need this?
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs

echo "nnodes: ${NNODES}"

## Vector's cluster doesn't support infinite bandwidth
## but gloo backend would automatically use inifinite bandwidth if not disable
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

export NCCL_DEBUG=INFO

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
echo SLURM_NTASKS=$SLURM_NTASKS

module purge
module add anaconda3/gpu/2022.10
module load anaconda3/gpu/2022.10
module load gcc/11.2.0
conda activate medsam
module purge
module load slurm/current

export WANDB_API_KEY='e51b9830c285698aced24214131f58ee2bfbd5e3'
export WANDB_CACHE_DIR='/gpfs/home/kn2347/.cache/wandb'
export WANDB_CONFIG_DIR='/gpfs/home/kn2347/.config/wandb'
export WANDB_DIR='/gpfs/home/kn2347/wandb'
#export CUDA_VISIBLE_DEVICES='0,1,2,3'
#export CUDA_VISIBLE_DEVICES='0'
export CUDA_VISIBLE_DEVICES='0,1'
export work_dir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/singletask_medsam_continue_ckpt_dice_loss_8-13-24'
for (( i=0; i < $SLURM_NTASKS; ++i ))
do
    srun -lN1 --mem=100G --gres=gpu:1 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python experiment_code/1_train_medsam/train_multi_gpus_modified_multiclass.py \
        --data_frame_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv \
        --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
        --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/class_mappings/label${SLURM_ARRAY_TASK_ID}_only_name_class_mapping.csv \
        -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
        -task_name singletask_medsam_no_yolo \
        --wandb_run_name label${SLURM_ARRAY_TASK_ID}_continue_dice \
        -checkpoint /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/medsam_vit_b.pth \
        --resume /gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/singletask_medsam_training_all_labels_8-7-24/lr5e-4/${SLURM_ARRAY_TASK_ID}/singletask_medsam_no_yolo-label${SLURM_ARRAY_TASK_ID}_training-*/medsam_model_best.pth \
        -work_dir ${work_dir}/${SLURM_ARRAY_TASK_ID} \
        -label_id 1 \
        -batch_size 256 \
        -num_workers 4 \
        -num_epochs 10 \
        --lambda_dice 1 \
        -lr 5e-4 \
        -use_wandb True \
        -use_amp \
        --suppress_train_debug_imgs \
        --world_size ${WORLD_SIZE} \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank ${i} \
        --init_method tcp://${MASTER_ADDR}:${MASTER_PORT}" >> ${work_dir}/label${SLURM_ARRAY_TASK_ID}.log 2>&1 &
done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"