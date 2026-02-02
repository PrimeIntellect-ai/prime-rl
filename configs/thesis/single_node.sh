#!/bin/bash

#SBATCH --job-name=thesis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive                 # Get the ENTIRE node exclusively
#SBATCH --output=/shared/mika/job_%j.log
#SBATCH --error=/shared/mika/job_%j.log

# Configs
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-""}

# Ensure experiment name is set
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "EXPERIMENT_NAME is not set"
    exit 1
fi

export BASE_DIR=${BASE_DIR:-"/home/mika/prime-rl"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/shared/mika/$EXPERIMENT_NAME"}
export TRAIN_CMD=${TRAIN_CMD:-"uv run rl @ configs/thesis/wiki_search.toml"}

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Cleanup
srun bash -c 'pkill -9 -f torchrun || true && fuser -k 29500/tcp || true'
srun bash -c 'pkill -9 -f VLLM || true && fuser -k 8000/tcp || true'

# Install environment
cd $BASE_DIR
source .env
source .venv/bin/activate

# Install environment as local package
uv sync --all-extras

# Run RL
srun bash -c '
    # Setup environment
    cd $BASE_DIR
    source .env
    source .venv/bin/activate

    $TRAIN_CMD
'
