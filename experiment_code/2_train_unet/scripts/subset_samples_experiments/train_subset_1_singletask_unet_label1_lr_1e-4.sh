#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=singletask_unet
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_short
#SBATCH --time=1-00:00:00
#SBATCH --array=1
#SBATCH --exclude=a100-4023,a100-4024,a100-4018,a100-4019

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
export CUDA_VISIBLE_DEVICES='0'
export pct_subsample=1
export work_dir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subset_experiments_singletask_unet_8-21-24'
mkdir -p ${work_dir}/${SLURM_ARRAY_TASK_ID}/${pct_subsample}
for (( i=0; i < $SLURM_NTASKS; ++i ))
do
    srun -lN1 --mem=100G --gres=gpu:1 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python experiment_code/2_train_unet/train_unet.py \
    --data_frame_path /gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df/baseline_unet/all_labels_df.csv \
    -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/subset_train_id_dfs_pooled/${pct_subsample}.pkl \
    --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv \
    -label_id ${SLURM_ARRAY_TASK_ID} -num_classes 1 -batch_size 64 -num_workers 2 \
    -lr .0001 \
    -epochs 50000 \
    -work_dir ${work_dir}/${SLURM_ARRAY_TASK_ID}/${pct_subsample} \
    --early_stop_delta 0.005 \
    --early_stop_patience 10 \
    -project_name singletask_unet -wandb_run_name label${SLURM_ARRAY_TASK_ID}_subsample_${pct_subsample}%_training" >> ${work_dir}/${SLURM_ARRAY_TASK_ID}/${pct_subsample}/log_${SLURM_JOBID}.log 2>&1 &
done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"