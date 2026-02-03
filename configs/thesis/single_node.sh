#!/bin/bash

#SBATCH --job-name=thesis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive                 # Get the ENTIRE node exclusively
#SBATCH --output=/shared/mika/job_%j.log
#SBATCH --error=/shared/mika/job_%j.log

set -euxo pipefail  # Debug: print commands, fail on error

echo "=== Job started at $(date) ==="
echo "=== Running on $(hostname) ==="
echo "=== Job ID: $SLURM_JOB_ID ==="

# Configs
export GIT_REF=${GIT_REF:-"thesis"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"job_${SLURM_JOB_ID}"}
export BASE_DIR=${BASE_DIR:-"/home/mika/prime-rl"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/shared/mika/$EXPERIMENT_NAME"}

echo "=== EXPERIMENT_NAME: $EXPERIMENT_NAME ==="
export TRAIN_CMD=${TRAIN_CMD:-"uv run rl @ configs/thesis/wiki_search.toml --output-dir $OUTPUT_DIR"}

echo "=== BASE_DIR: $BASE_DIR ==="
echo "=== OUTPUT_DIR: $OUTPUT_DIR ==="
echo "=== TRAIN_CMD: $TRAIN_CMD ==="

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

# Run RL
echo "=== Starting RL training ==="
srun bash -c '
    set -euxo pipefail
    # Setup environment
    cd $BASE_DIR
    git checkout $GIT_REF
    source /home/mika/.env
    source .venv/bin/activate

    $TRAIN_CMD --output-dir $OUTPUT_DIR
'

echo "=== Job finished at $(date) ==="
