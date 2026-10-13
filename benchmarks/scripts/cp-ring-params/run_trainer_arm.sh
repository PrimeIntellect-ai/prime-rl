#!/bin/bash
# Run one fake-data RL trainer arm (no inference) on samsja/mini-glm-moe with packed documents.
# usage: run_trainer_arm.sh <worktree> <name> <gpus> <nproc> <master-port> [extra trainer args...]
set -u
WT=$1; NAME=$2; GPUS=$3; NPROC=$4; PORT=$5; shift 5
OUT=${CP_RING_OUT:-$HOME/tmp/cp-ring-params/runs}/$NAME
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$WT"
git rev-parse HEAD > "$OUT/sha"
export CUDA_VISIBLE_DEVICES=$GPUS
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-$USER-cpring-$NAME
rm -rf "$TORCHINDUCTOR_CACHE_DIR"
export TORCH_LOGS="recompiles,graph_breaks"
export TORCH_LOGS_FORMAT="[%(threadName)s] %(name)s: %(message)s"
start=$(date +%s)
uv run --no-sync torchrun --nproc-per-node="$NPROC" --master-port="$PORT" src/prime_rl/trainer/rl/train.py \
  @ configs/debug/fake/rl.toml --model.name samsja/mini-glm-moe --model.seq-len 2048 --max-steps 20 \
  --data.fake.generate-samples --data.fake.batch-size 8 --output-dir "$OUT/run" "$@" > "$OUT/trainer.log" 2>&1
echo "exit=$? wall_s=$(( $(date +%s) - start ))" >> "$OUT/trainer.log"
