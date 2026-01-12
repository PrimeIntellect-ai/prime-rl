#!/bin/bash

#SBATCH --job-name=daniel-repro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive                 # Get the ENTIRE node exclusively
#SBATCH --output=/shared/logs/job_%j.log
#SBATCH --error=/shared/logs/job_%j.log

# Configs
export CKPT_STEP=-1

# Paths
export BASE_DIR=${BASE_DIR:-"/home/daniel/prime-rl3"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/home/daniel/outputs/$SLURM_JOB_NAME"}
mkdir -p $OUTPUT_DIR/slurm

# Clear previous weights and broadcast flags
# We delete rollout dir to prevent the trainer from training on rollouts from a previous run.
# We delete broadcast flags to prevent the orchestrator from prematurely notifying the inference to receive weights
# WARN: This is potentially dangerous and could lead to data loss
rm -rf $OUTPUT_DIR/rollouts
rm -rf $OUTPUT_DIR/broadcasts
rm -rf $OUTPUT_DIR/run_0/rollouts
rm -rf $OUTPUT_DIR/run_0/broadcasts

# Remove the STABLE files in weight directory to respect the async barrier on orchestrator
# rm $OUTPUT_DIR/weights/*/STABLE

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Cleanup
srun bash -c 'pkill -9 -f prime-rl || true'
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

    # Disable sync to avoid conflicts with lockfile

    # Higher ulimit
    ulimit -n 65536
    export GIT_LFS_SKIP_SMUDGE=1

    # Infiniband setup
    IB_HCA=$(ibv_devinfo | sed -n -e '/hca_id/p' -e '/link_layer:/p' | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' | tr -d '[[:blank:]]' |paste -sd,)
    export NCCL_IB_HCA=$IB_HCA

    # This is required for vLLM graph compile to work
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1


    
    # This is required for compilation to work correctly
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    uv run rl @ configs/alphabet_sort/rl.toml \
        --wandb.project "daniel-repro" \
        --wandb.name "alphabet-sort" \
        --output-dir $OUTPUT_DIR \
        2>&1 | tee $OUTPUT_DIR/slurm/latest_orchestrator.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_orchestrator.log

'
