#!/bin/bash

#SBATCH --job-name=thesis
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/shared/mika/job_%j.log
#SBATCH --error=/shared/mika/job_%j.log

set -euxo pipefail  # Debug: print commands, fail on error

echo "=== Job started at $(date) ==="
echo "=== Running on $(hostname) ==="
echo "=== Job ID: $SLURM_JOB_ID ==="

# Configs
export GIT_REF=${GIT_REF:-"thesis"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"job_${SLURM_JOB_ID}"}
export TRAIN_ARGS=${TRAIN_ARGS:-""}
export BASE_DIR=${BASE_DIR:-"/home/mika/prime-rl"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/shared/mika/$EXPERIMENT_NAME"}

echo "=== EXPERIMENT_NAME: $EXPERIMENT_NAME ==="
echo "=== TRAIN_ARGS: $TRAIN_ARGS ==="
echo "=== BASE_DIR: $BASE_DIR ==="
echo "=== OUTPUT_DIR: $OUTPUT_DIR ==="

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Cleanup (ignore errors)
echo "=== Cleanup ==="
srun bash -c 'pkill -9 -f torchrun || true && fuser -k 29500/tcp || true' || true
srun bash -c 'pkill -9 -f VLLM || true && fuser -k 8000/tcp || true' || true

# Install environment
echo "=== Setting up environment ==="
cd $BASE_DIR
git checkout $GIT_REF
source /home/mika/.env
source .venv/bin/activate

# Install environment as local package
echo "=== Running uv sync ==="
uv sync --all-extras

export NUM_TRAIN_NODES=${NUM_TRAIN_NODES:-1}
export NUM_INFER_NODES=${NUM_INFER_NODES:-1}

if [ $((NUM_TRAIN_NODES + NUM_INFER_NODES)) != $SLURM_JOB_NUM_NODES ]; then
    echo "NUM_TRAIN_NODES + NUM_INFER_NODES must equal SLURM_JOB_NUM_NODES"
    exit 1
fi

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Networking
export HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export INFER_HOSTS=${HOSTNAMES[@]:0:$NUM_INFER_NODES}
export TRAIN_HOSTS=${HOSTNAMES[@]:$NUM_INFER_NODES:$SLURM_JOB_NUM_NODES}

INFER_URLS=""
for host in ${INFER_HOSTS[@]}; do
    if [ -z "$INFER_URLS" ]; then
        INFER_URLS="http://$host:8000/v1"
    else
        INFER_URLS="$INFER_URLS,http://$host:8000/v1"
    fi
done
export INFER_URLS
echo "HOSTNAMES=${HOSTNAMES[@]}"
echo "TRAIN_HOSTS=${TRAIN_HOSTS[@]}"
echo "INFER_HOSTS=${INFER_HOSTS[@]}"
echo "INFER_URLS=${INFER_URLS}"

export MASTER_ADDR="${HOSTNAMES[$NUM_INFER_NODES]}"
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Create output directories
mkdir -p $OUTPUT_DIR/slurm $OUTPUT_DIR/torchrun

# Run RL
srun bash -c '
    # Setup environment
    cd $BASE_DIR
    source /home/mika/.env
    source .venv/bin/activate

    # Higher ulimit
    ulimit -n 65536
    export GIT_LFS_SKIP_SMUDGE=1

if [ "$SLURM_PROCID" -lt "$NUM_INFER_NODES" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1

    # This is required for vLLM graph compile to work
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
    INFER_NODE_RANK=$SLURM_PROCID
        echo $HOSTNAMES | tee  $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log
    uv run inference \
        @ configs/thesis/deepdive/infer.toml \
        2>&1 | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${INFER_NODE_RANK}_infer_node_rank_${INFER_NODE_RANK}.log
else
    TRAIN_NODE_RANK=$((SLURM_PROCID - NUM_INFER_NODES))

    if [ "$TRAIN_NODE_RANK" -eq 0 ]; then
        uv run orchestrator \
            @ configs/thesis/deepdive/orch.toml \
            --client.base-url $INFER_URLS \
            --client.timeout 3600 \
            --num-train-workers $((NUM_TRAIN_NODES * 8)) \
            --output-dir $OUTPUT_DIR/run_0 \
            2>&1 | tee $OUTPUT_DIR/slurm/latest_orchestrator.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_orchestrator.log & disown
    fi

    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    
    # This is required for compilation to work correctly
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    echo $HOSTNAMES | tee  $OUTPUT_DIR/slurm/latest_train_node_rank_${TRAIN_NODE_RANK}.log
    uv run torchrun \
        --nnodes=$NUM_TRAIN_NODES \
        --nproc-per-node=8 \
        --node-rank=$TRAIN_NODE_RANK \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv-id=job_$SLURM_JOB_ID \
        --log-dir=$OUTPUT_DIR/torchrun \
        --tee=3 \
        --redirects=3 \
        --local-ranks-filter=0 \
        src/prime_rl/trainer/rl/train.py \
        @ configs/thesis/deepdive/train.toml \
        --output-dir $OUTPUT_DIR \
        $TRAIN_ARGS \
        2>&1 | tee -a $OUTPUT_DIR/slurm/latest_train_node_rank_${TRAIN_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_train_node_rank_${TRAIN_NODE_RANK}.log
fi
'
