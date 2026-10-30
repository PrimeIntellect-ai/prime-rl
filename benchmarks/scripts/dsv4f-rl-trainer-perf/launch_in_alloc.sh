#!/bin/bash
# Renders an `rl` launch with --dry-run and runs its sbatch script inside an existing salloc
# allocation. Fresh compile caches per run.
# Usage: MODEL=<hf id> launch_in_alloc.sh <jobid> <launch worktree> <run name> [rl args...]
set -euo pipefail
: "${MODEL:?set MODEL to the checkpoint to benchmark}"
JOBID=$1
LAUNCH_DIR=$2
RUN_NAME=$3
shift 3

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=${OUT_DIR:-$HOME/outputs/dsv4f-rl-trainer-perf}

cd "$LAUNCH_DIR"
uv run --no-sync rl @ "$SCRIPT_DIR/proxy.toml" \
    --model.name "$MODEL" \
    --output-dir "$OUT_DIR" \
    --run.name "$RUN_NAME" \
    --env-vars.TILELANG_CACHE_DIR "/tmp/garrett/$RUN_NAME/tilelang" \
    --env-vars.TRITON_CACHE_DIR "/tmp/garrett/$RUN_NAME/triton" \
    --env-vars.TORCHINDUCTOR_CACHE_DIR "/tmp/garrett/$RUN_NAME/inductor" \
    --env-vars.DG_JIT_CACHE_DIR "/tmp/garrett/$RUN_NAME/deep-gemm" \
    --env-vars.CUDA_CACHE_PATH "/tmp/garrett/$RUN_NAME/cuda" \
    --env-vars.TORCH_LOGS "recompiles,graph_breaks" \
    --dry-run "$@"

SBATCH_SCRIPT=$OUT_DIR/$RUN_NAME/launcher/rl.sbatch
LAUNCHER_DIR=$(dirname "$SBATCH_SCRIPT")
echo "commit $(git -C "$LAUNCH_DIR" rev-parse HEAD)" > "$LAUNCHER_DIR/code_commit.txt"
echo "Running $SBATCH_SCRIPT in allocation $JOBID"

export SLURM_JOB_ID=$JOBID
export SLURM_JOB_NUM_NODES=$(squeue -h -j "$JOBID" -o %D)
export SLURM_NNODES=$SLURM_JOB_NUM_NODES
export SLURM_NTASKS=$SLURM_JOB_NUM_NODES
export SLURM_NTASKS_PER_NODE=1
export SLURM_OVERLAP=1
# Steps number their tasks in the allocation's topology order, not the sorted node list, so list
# the nodes by task id: the script derives MASTER_ADDR and node ranks from this order.
export SLURM_JOB_NODELIST=$(srun bash -c 'echo $SLURM_PROCID $(hostname)' | sort -n | awk '{print $2}' | paste -sd,)
echo "Nodes by task id: $SLURM_JOB_NODELIST"
bash "$SBATCH_SCRIPT" > "$LAUNCHER_DIR/job_alloc_$JOBID.log" 2>&1
