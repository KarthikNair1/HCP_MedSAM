#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=n-5nodes
#SBATCH --mem=200GB
#SBATCH --gres=gpu:2
#SBATCH --partition=a100_dev
#SBATCH --output=logs/medsam_mgpus_%x-%j.out
#SBATCH --error=logs/medsam_mgpus_%x-%j.err
#SBATCH --time=04:00:00
###SBATCH --exclude=a100-4023,a100-4024,a100-4018,a100-4019


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
for (( i=0; i < $SLURM_NTASKS; ++i ))
do
    srun -lN1 --mem=200G --gres=gpu:2 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python /gpfs/home/kn2347/MedSAM/train_multi_gpus_modified_multiclass.py \
        --data_frame_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv \
        -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
        -class_weights_dict_path /gpfs/data/luilab/karthik/pediatric_seg_proj/class_weights_256.pkl \
        -task_name MedSAM_finetune_hcp_ya_constant_bbox_all_tasks \
        -checkpoint /gpfs/home/kn2347/MedSAM/medsam_vit_b.pth \
        -work_dir /gpfs/home/kn2347/loss_reweighted_results_7-31-23 \
        -batch_size 32 \
        -num_workers 4 \
        -num_epochs 150 \
        --lambda_dice 0.001 \
        -lr 1e-4 \
        --loss_reweighted \
        -use_wandb True \
        --world_size ${WORLD_SIZE} \
        --bucket_cap_mb 25 \
        --grad_acc_steps 1 \
        --node_rank ${i} \
        --init_method tcp://${MASTER_ADDR}:${MASTER_PORT}" >> ./logs/log_for_${SLURM_JOB_ID}_node_${i}.log 2>&1 &
done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"