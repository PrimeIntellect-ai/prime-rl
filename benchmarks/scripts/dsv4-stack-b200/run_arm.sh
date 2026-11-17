#!/usr/bin/env bash
# Run one arm of the DSv4 one-node trainer-only proxy inside an existing SLURM allocation.
# Usage: run_arm.sh <jobid> <run name> [trainer CLI overrides...]
# The run dir mirrors the rl/sft launcher layout so the profiling scripts can read it.
set -euo pipefail

JOBID=$1
RUN=$2
shift 2

WORKTREE=$(cd "$(dirname "$0")/../../.." && pwd)
RUN_DIR=${OUT_ROOT:-$HOME/prl_output_dir/dsv4-perf-proxy}/$RUN
LOG_DIR=$RUN_DIR/logs/attempt_1
CACHE_DIR=/tmp/$USER/dsv4-perf-proxy/$RUN
if [[ -e $RUN_DIR ]]; then
    echo "Run dir $RUN_DIR already exists" >&2
    exit 1
fi
mkdir -p "$LOG_DIR/trainer"
git -C "$WORKTREE" rev-parse HEAD > "$RUN_DIR/commit.txt"
git -C "$WORKTREE" status --short > "$RUN_DIR/git_status.txt"
OVERRIDES=""
if (($# > 0)); then
    OVERRIDES=$(printf '%q ' "$@")
fi
echo "$OVERRIDES" > "$RUN_DIR/overrides.txt"

srun --jobid="$JOBID" --overlap -N1 -n1 --gres=gpu:8 bash -c "
set -euo pipefail
cd $WORKTREE
rm -rf $CACHE_DIR
mkdir -p $CACHE_DIR/{inductor,triton,tilelang,deep_gemm}
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR/inductor TRITON_CACHE_DIR=$CACHE_DIR/triton
export TILELANG_CACHE_DIR=$CACHE_DIR/tilelang DG_JIT_CACHE_DIR=$CACHE_DIR/deep_gemm
export TORCH_LOGS=recompiles,graph_breaks
nvidia-smi --query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,memory.used --format=csv -l 5 \
    > $RUN_DIR/nvidia_smi.csv 2>&1 &
SMI_PID=\$!
status=0
uv run --no-sync torchrun --nproc-per-node=8 --log-dir=$LOG_DIR/trainer/torchrun --redirects=3 --tee=3 \
    --local-ranks-filter=0 -m prime_rl.trainer.rl.train @ benchmarks/scripts/dsv4-perf-proxy/proxy.toml \
    --output-dir $RUN_DIR $OVERRIDES > $LOG_DIR/trainer.log 2>&1 || status=\$?
kill \$SMI_PID || true
find $CACHE_DIR/deep_gemm/cache -mindepth 1 -maxdepth 1 -printf '%T@ %f\n' \
    > $LOG_DIR/deep_gemm_cache_\$(hostname).txt 2>/dev/null || true
exit \$status
"
