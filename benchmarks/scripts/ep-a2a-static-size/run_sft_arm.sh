#!/bin/bash
# Run one fake-data SFT arm with variable-length packed documents.
# usage: run_sft_arm.sh <worktree> <name> <gpus> <num-gpus> [extra sft args...]
set -u
WT=$1; NAME=$2; GPUS=$3; NGPU=$4; shift 4
OUT=${EP_A2A_OUT:-$HOME/tmp/ep-a2a-static-size/runs}/$NAME
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$WT"
git rev-parse HEAD > "$OUT/sha"
export CUDA_VISIBLE_DEVICES=$GPUS
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/ep-a2a-$NAME/inductor
export TRITON_CACHE_DIR=/tmp/$USER/ep-a2a-$NAME/triton
rm -rf "/tmp/$USER/ep-a2a-$NAME"
export TORCH_LOGS="recompiles,graph_breaks"
export TORCH_LOGS_FORMAT="[%(threadName)s] %(name)s: %(message)s"
export PRL_OUTPUT_DIR=$OUT
start=$(date +%s)
uv run --no-sync sft @ configs/debug/fake/sft.toml --output-dir "$OUT/run" --deployment.num-train-gpus "$NGPU" \
  --data.length variable "$@" > "$OUT/sft.log" 2>&1
echo "exit=$? wall_s=$(( $(date +%s) - start ))" >> "$OUT/sft.log"
