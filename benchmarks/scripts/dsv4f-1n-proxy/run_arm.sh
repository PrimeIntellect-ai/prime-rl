#!/bin/bash
# Run one arm of the one-node proxy with torchrun inside an allocated node, on the code of CODE_DIR
# (default: this worktree) and the venv of this worktree. Fresh compile caches per arm.
# usage: MODEL=<hf id> JOBID=<slurm job> [CODE_DIR=<worktree>] run_arm.sh <arm-name> [trainer CLI overrides...]
set -euo pipefail
: "${JOBID:?set JOBID to the slurm job id of the allocated node}"
: "${MODEL:?set MODEL to the checkpoint to benchmark}"
ARM=$1
shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CODE_DIR=${CODE_DIR:-$REPO}
RUNS_DIR=${RUNS_DIR:-$HOME/outputs/dsv4f-1n-proxy}
OUT=$RUNS_DIR/$ARM
rm -rf "$OUT"
mkdir -p "$OUT"
printf '%q ' "$0" "$ARM" "$@" >"$OUT/command.txt"
git -C "$CODE_DIR" log -1 --format=%H >"$OUT/commit.txt"
git -C "$CODE_DIR" diff HEAD >"$OUT/uncommitted.diff"
export REPO CODE_DIR OUT ARM MODEL
EXTRA=""
[ $# -gt 0 ] && EXTRA=$(printf '%q ' "$@")
export EXTRA

srun --jobid "$JOBID" -N1 -n1 --gres=gpu:8 --overlap --kill-on-bad-exit=1 bash -c '
set -euo pipefail
cd "$REPO"
CACHE=/tmp/$USER/dsv4f-1n-proxy/$ARM
rm -rf "$CACHE"
mkdir -p "$CACHE"
export PYTHONPATH=$CODE_DIR/src:$CODE_DIR/packages/prime-rl-configs/src
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export HF_HUB_CACHE=/home/huggingface/hub HF_XET_CACHE=/home/huggingface/xet HF_DATASETS_CACHE=/home/huggingface/datasets HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMPY_MADVISE_HUGEPAGE=0
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton TILELANG_CACHE_DIR=$CACHE/tilelang
export DG_JIT_CACHE_DIR=$CACHE/deep_gemm CUDA_CACHE_PATH=$CACHE/cuda
export TORCH_LOGS=recompiles,graph_breaks
ulimit -n 65536
hostname >"$OUT/host.txt"
status=0
eval uv run --no-sync torchrun --nproc-per-node=8 --log-dir="$OUT/torchrun" --redirects=3 --tee=3 \
    --local-ranks-filter=0 -m prime_rl.trainer.rl.train \
    @ benchmarks/scripts/dsv4f-1n-proxy/proxy.toml --model.name "$MODEL" --output-dir "$OUT" $EXTRA 2>&1 | tee "$OUT/node.log" || status=$?
exit $status
'
