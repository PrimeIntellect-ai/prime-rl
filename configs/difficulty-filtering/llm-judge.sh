#!/bin/bash

#SBATCH --job-name=vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive                 # Get the ENTIRE node exclusively
#SBATCH --output=/shared/logs/job_%j.log
#SBATCH --error=/shared/logs/job_%j.log

# Configs
export MODEL_NAME=${MODEL_NAME:-"opencompass/CompassVerifier-3B"}
export DP=${DP:-8}
export TP=${TP:-1}
export IFNAME=${IFNAME:-"bond0"}

# Update job name to include model slug
export MODEL_ORG=$(echo "$MODEL_NAME" | cut -d'/' -f1)
export MODEL_ID=$(echo "$MODEL_NAME" | cut -d'/' -f2)
export JOB_NAME="$(echo $MODEL_ID | tr '[:upper:]' '[:lower:]')-vllm"
scontrol update JobId=$SLURM_JOB_ID JobName="${JOB_NAME}" 2>/dev/null || true

# Change directory
export BASE_DIR=${BASE_DIR:-"/home/mika/prime-rl"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/home/mika/outputs/$SLURM_JOB_NAME"}
mkdir -p $OUTPUT_DIR/slurm

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Cleanup
srun bash -c "sudo pkill -9 -f VLLM || true && sudo fuser -k 8000/tcp || true"

# Run inference
srun bash -c '
    cd $BASE_DIR
    source home/mika/.env
    source .venv/bin/activate

    # Infiniband setup
    IB_HCA=$(ibv_devinfo | sed -n -e '/hca_id/p' -e '/link_layer:/p' | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' | tr -d '[[:blank:]]' |paste -sd,)
    export NCCL_IB_HCA=$IB_HCA

    # Pipe IP addresses to log file
    LOCAL_IP=$(ip -o -4 addr show dev $IFNAME | awk "{print \$4}" | cut -d/ -f1)
    PUBLIC_IP=$(curl -s ipinfo.io/ip)
    echo "LOCAL_IP=${LOCAL_IP}, PUBLIC_IP=${PUBLIC_IP}" | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${SLURM_PROCID}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_infer_node_rank_${SLURM_PROCID}.log

    # Required for graph compilation to work
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

    # Start inference
    vllm serve $MODEL_NAME \
    --tensor-parallel-size $TP \
    --data-parallel-size $DP \
    2>&1 | tee -a $OUTPUT_DIR/slurm/latest_infer_node_rank_${SLURM_PROCID}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_infer_node_rank_${SLURM_PROCID}.log
'
