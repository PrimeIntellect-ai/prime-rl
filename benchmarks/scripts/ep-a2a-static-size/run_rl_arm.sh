#!/bin/bash
# Run one RL arm of the EP all-to-all matrix (2 trainer GPUs, 1 inference GPU).
# EP_A2A_RL_CONFIG overrides the config, which defaults to reverse-text-moe.
# usage: run_rl_arm.sh <worktree> <name> <gpus> <steps> [extra rl args...]
set -u
WT=$1; NAME=$2; GPUS=$3; STEPS=$4; shift 4
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
uv run --no-sync rl @ "${EP_A2A_RL_CONFIG:-configs/ci/integration/reverse-text-moe/start.toml}" \
  --output-dir "$OUT/run" --max-steps "$STEPS" --ckpt None \
  --deployment.num-train-gpus 2 --deployment.num-infer-gpus 1 "$@" > "$OUT/rl.log" 2>&1
echo "exit=$? wall_s=$(( $(date +%s) - start ))" >> "$OUT/rl.log"
