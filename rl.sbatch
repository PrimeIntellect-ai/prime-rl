#!/bin/bash

#SBATCH --job-name=test-pp-multinode-e2e
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/home/matej/dev/prime-rl/logs/%x-%j.out
#SBATCH --error=/home/matej/dev/prime-rl/logs/%x-%j.err

# Configs
export NUM_TRAIN_NODES=1
export NUM_INFER_NODES=2
export GPUS_PER_NODE=8
export INFERENCE_TP=8
export INFERENCE_PP=2
export INFERENCE_ENABLE_EXPERT_PARALLEL=0
export INFERENCE_DATA_PARALLEL_RPC_PORT=13345
export INFERENCE_DP_LOCAL=$((GPUS_PER_NODE / INFERENCE_TP))

# Enable distributed EP mode only when expert parallelism spans multiple inference nodes.
export USE_DISTRIBUTED_EP=0
if [ "$INFERENCE_ENABLE_EXPERT_PARALLEL" -eq 1 ] && [ "$NUM_INFER_NODES" -gt 1 ]; then
    USE_DISTRIBUTED_EP=1
fi

# Enable multi-node PP mode when pipeline parallelism spans multiple inference nodes.
export USE_MULTI_NODE_PP=0
if [ "$INFERENCE_PP" -gt 1 ] && [ "$NUM_INFER_NODES" -gt 1 ]; then
    USE_MULTI_NODE_PP=1
fi

# Paths
export PROJECT_DIR=/home/matej/dev/prime-rl
export CONFIG_DIR=/home/matej/dev/prime-rl/configs/test_pp_multinode
export OUTPUT_DIR=/home/matej/dev/prime-rl/outputs/test_pp_multinode
export ORCHESTRATOR_OUTPUT_DIR=/home/matej/dev/prime-rl/outputs/test_pp_multinode/run_default
mkdir -p $OUTPUT_DIR/slurm

# Clear previous rollouts and broadcast flags from ALL runs (not just the current one)
# to prevent the trainer from training on stale data from old run_* directories.
# WARN: This is potentially dangerous and could lead to data loss
rm -rf $OUTPUT_DIR/run_*/rollouts
rm -rf $OUTPUT_DIR/run_*/broadcasts
rm -rf $OUTPUT_DIR/rollouts

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Networking
HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
INFER_HOSTS=( ${HOSTNAMES[@]:0:$NUM_INFER_NODES} )
TRAIN_HOSTS=( ${HOSTNAMES[@]:$NUM_INFER_NODES:$SLURM_JOB_NUM_NODES} )
export INFER_HEAD_HOST="${INFER_HOSTS[0]}"
export HOSTNAMES_STR="${HOSTNAMES[*]}"

INFER_URLS=""
if [ "$USE_DISTRIBUTED_EP" -eq 1 ] || [ "$USE_MULTI_NODE_PP" -eq 1 ]; then
    # Single URL: only the head inference node serves the API
    INFER_URLS="http://${INFER_HEAD_HOST}:8000/v1"
else
    for host in "${INFER_HOSTS[@]}"; do
        if [ -z "$INFER_URLS" ]; then
            INFER_URLS="http://$host:8000/v1"
        else
            INFER_URLS="$INFER_URLS,http://$host:8000/v1"
        fi
    done
fi
export INFER_URLS
echo "HOSTNAMES=${HOSTNAMES[*]}"
echo "TRAIN_HOSTS=${TRAIN_HOSTS[*]}"
echo "INFER_HOSTS=${INFER_HOSTS[*]}"
echo "INFER_URLS=${INFER_URLS}"
echo "USE_DISTRIBUTED_EP=${USE_DISTRIBUTED_EP}"

export MASTER_ADDR="${HOSTNAMES[$NUM_INFER_NODES]}"
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Setup environment
cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate
uv sync --all-extras

# Run RL
srun bash -c '
    # Source environment
    cd $PROJECT_DIR
    [ -f .env ] && source .env
    source .venv/bin/activate
    # Do not sync to avoid conflicts with lockfile

    # Higher ulimit
    ulimit -n 65536
    export GIT_LFS_SKIP_SMUDGE=1

    # Infiniband setup
    IB_HCA=$(ibv_devinfo | sed -n -e "/hca_id/p" -e "/link_layer:/p" | grep -B1 InfiniBand | grep hca_id | sed -e "s/^hca_id://g" | tr -d "[[:blank:]]" |paste -sd,)
    export NCCL_IB_HCA=$IB_HCA

if [ "$SLURM_PROCID" -lt "$NUM_INFER_NODES" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
    INFER_NODE_RANK=$SLURM_PROCID
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    if [ "$USE_DISTRIBUTED_EP" -eq 1 ]; then
        INFER_DP_START_RANK=$((INFER_NODE_RANK * INFERENCE_DP_LOCAL))
        if [ "$INFER_NODE_RANK" -eq 0 ]; then
            uv run inference \
                @ $CONFIG_DIR/inference.toml \
                --data-parallel-address $INFER_HEAD_HOST \
                --data-parallel-rpc-port $INFERENCE_DATA_PARALLEL_RPC_PORT \
                2>&1 | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_infer_node_rank_${INFER_NODE_RANK}.log
        else
            uv run inference \
                @ $CONFIG_DIR/inference.toml \
                --data-parallel-start-rank $INFER_DP_START_RANK \
                --data-parallel-address $INFER_HEAD_HOST \
                --data-parallel-rpc-port $INFERENCE_DATA_PARALLEL_RPC_PORT \
                --headless \
                2>&1 | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_infer_node_rank_${INFER_NODE_RANK}.log
        fi
    elif [ "$USE_MULTI_NODE_PP" -eq 1 ]; then
        if [ "$INFER_NODE_RANK" -eq 0 ]; then
            uv run inference \
                @ $CONFIG_DIR/inference.toml \
                --nnodes $NUM_INFER_NODES \
                --node-rank 0 \
                --master-addr $INFER_HEAD_HOST \
                2>&1 | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_infer_node_rank_${INFER_NODE_RANK}.log
        else
            uv run inference \
                @ $CONFIG_DIR/inference.toml \
                --nnodes $NUM_INFER_NODES \
                --node-rank $INFER_NODE_RANK \
                --master-addr $INFER_HEAD_HOST \
                --headless \
                2>&1 | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_infer_node_rank_${INFER_NODE_RANK}.log
        fi
    else
        uv run inference \
            @ $CONFIG_DIR/inference.toml \
            2>&1 | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_infer_node_rank_${INFER_NODE_RANK}.log
    fi

else
    TRAIN_NODE_RANK=$((SLURM_PROCID - NUM_INFER_NODES))

    if [ "$TRAIN_NODE_RANK" -eq 0 ]; then
        uv run orchestrator \
            @ $CONFIG_DIR/orchestrator.toml \
            --weight_broadcast.host $MASTER_ADDR \
            --client.base-url $INFER_URLS \
            2>&1 | tee $OUTPUT_DIR/slurm/latest_orchestrator.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_orchestrator.log &
    fi

    # This is required for compilation to work correctly
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    echo $HOSTNAMES_STR | tee $OUTPUT_DIR/slurm/latest_train_node_rank_${TRAIN_NODE_RANK}.log

    uv run torchrun \
        --nnodes=$NUM_TRAIN_NODES \
        --nproc-per-node=$GPUS_PER_NODE \
        --node-rank=$TRAIN_NODE_RANK \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv-id=job_$SLURM_JOB_ID \
        --log-dir=$OUTPUT_DIR/torchrun \
        --tee=3 \
        --redirects=3 \
        --local-ranks-filter=$(seq -s, 0 $((GPUS_PER_NODE - 1))) \
        -m prime_rl.trainer.rl.train \
        @ $CONFIG_DIR/trainer.toml \
        2>&1 | tee -a $OUTPUT_DIR/slurm/latest_train_node_rank_${TRAIN_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_train_node_rank_${TRAIN_NODE_RANK}.log
    fi
'
