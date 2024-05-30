#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=10
#SBATCH --job-name=train_3d
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short

#SBATCH --output=logs3/medsam_mgpus_%x-%j.out
#SBATCH --error=logs3/medsam_mgpus_%x-%j.err
#SBATCH --time=3-00:00:00
###SBATCH --exclude=a100-4023,a100-4024,a100-4018,a100-4019,a100-4011


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
export CUDA_VISIBLE_DEVICES='0'
#export CUDA_VISIBLE_DEVICES='0,1'
num_tasks=$SLURM_NTASKS
for (( i=0; i < num_tasks; ++i ))
do
    srun -lN1 --mem=100G --gres=gpu:1 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python /gpfs/home/kn2347/MedSAM/train_multi_gpus_modified_multiclass_3D.py \
        --data_frame_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_with_yolov7_box_path.csv \
        --df_desired_path /gpfs/home/kn2347/MedSAM/darts_name_class_mapping_processed.csv \
        -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
        -task_name MedSAM_finetune_3D \
        --wandb_run_name yolov7_3D \
        -checkpoint /gpfs/home/kn2347/MedSAM/medsam_vit_b.pth \
        -work_dir /gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/3d_model_8-31-23 \
        -batch_size 1 \
        -num_workers 8 \
        -num_epochs 60 \
        --lambda_dice 1 \
        -lr 1e-4 \
        -use_wandb True \
        -use_amp \
        --suppress_train_debug_imgs \
        --world_size ${WORLD_SIZE} \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank ${i} \
        --resume /gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/3d_model_8-31-23/MedSAM_finetune_3D-20230902-230440/medsam_model_latest.pth \
        --init_method tcp://${MASTER_ADDR}:${MASTER_PORT}" >> ./logs3/log_for_${SLURM_JOB_ID}_node_${i}.log 2>&1 &
done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"