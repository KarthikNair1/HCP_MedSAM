#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=singletask_medsam_oracle_test
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu8_medium
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --exclude=a100-4020
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


#export workdir='/gpfs/data/luilab/karthik/pediatric_seg_proj/results_copied_from_kn2347/subsets_expt_unet_9-12-24/training'

export label=$1
export frac=$2
export lr=$3
export workdir=$4
export repeat=$SLURM_ARRAY_TASK_ID

mkdir -p ${workdir}/test/${lr}/${label}/${frac}/${repeat}/at_least_100
if ls ${workdir}/test/${lr}/${label}/${frac}/${repeat}/at_least_100/*.pkl 1> /dev/null 2>&1; then 
    exit 1
fi

for (( i=0; i < $SLURM_NTASKS; ++i ))
do
    
    srun -lN1 --mem=16G -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python experiment_code/4_evaluate/full_eval_val_test.py \
    --model_type singletask_yolov7_prompted \
    --explicit_model_path "${workdir}/training/${lr}/${label}/${frac}/${repeat}/*/medsam_model_best.pth" \
    --explicit_yolo_bbox_dataframe_path /gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_isolated_df_new/${label}/0.1/isolated_path_df_bboxes_from_ground_truth.csv \
    --explicit_dataframe_to_merge_with_yolo_path /gpfs/data/luilab/karthik/pediatric_seg_proj/path_df_constant_bbox.csv \
    --dataframe_mask_npy_path /gpfs/data/luilab/karthik/pediatric_seg_proj/per_class_eval_dfs/at_least_100_mask_arrays/label${label}.npy \
    --label ${label} \
    --tag test \
    -train_test_splits /gpfs/data/luilab/karthik/pediatric_seg_proj/train_val_test_split.pickle \
    --batch_size 64 \
    --df_starting_mapping_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/hcp_mapping_processed.csv \
    --df_desired_path /gpfs/home/kn2347/HCP_MedSAM_project/modified_medsam_repo/darts_name_class_mapping_processed.csv \
    --output_dir ${workdir}/test/${lr}/${label}/${frac}/${repeat}/at_least_100" >> ${workdir}/test/${lr}/${label}/${frac}/${repeat}/at_least_100/log_for_${SLURM_JOB_ID}.log 2>&1 &
done
wait ## Wait for the tasks on nodes to finish

echo "END TIME: $(date)"