#!/bin/bash
# Run one end-to-end SFT arm on a full 8-GPU node: ep=8 (EP_A2A_EP overrides), bf16 router, no offloading,
# compile on.
# Workloads: qwen3-30b (fake data, balanced routing) or glm-air-8l (8 real GLM-4.5-Air layers, real SFT data).
# usage: run_e2e_arm.sh <worktree> <name> <workload> <max-steps> [extra sft args...]
set -uf
WT=$1; NAME=$2; WORKLOAD=$3; STEPS=$4; shift 4
OUT=${EP_A2A_E2E_OUT:-$HOME/tmp/ep-a2a-static-size/e2e}/$NAME
GLM_4_5_AIR=/home/huggingface/hub/models--zai-org--GLM-4.5-Air/snapshots/a24ceef6ce4f3536971efe9b778bdaa1bab18daa
case $WORKLOAD in
  qwen3-30b) MODEL="--model.name Qwen/Qwen3-30B-A3B-Instruct-2507 --model.debug.force-balanced-routing
    --data.type fake" ;;
  glm-air-8l) MODEL="--model.name $GLM_4_5_AIR --model.debug.num-layers 8 --renderer.name glm-4.5 --data.type sft
    --data.name PrimeIntellect/INTELLECT-3-SFT-10K --data.splits [\"math\"] --data.seed 0 --optim.lr 1e-5" ;;
  *) echo "unknown workload $WORKLOAD"; exit 1 ;;
esac
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$WT"
git rev-parse HEAD > "$OUT/sha"
unset HF_HOME
export HF_HUB_CACHE=/home/huggingface/hub
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/ep-a2a-e2e-$NAME/inductor
export TRITON_CACHE_DIR=/tmp/$USER/ep-a2a-e2e-$NAME/triton
rm -rf "/tmp/$USER/ep-a2a-e2e-$NAME"
export TORCH_LOGS="recompiles,graph_breaks"
export TORCH_LOGS_FORMAT="[%(threadName)s] %(name)s: %(message)s"
export PRL_OUTPUT_DIR=$OUT
start=$(date +%s)
uv run --no-sync sft $MODEL --output-dir "$OUT/run" --run.name bench --deployment.num-train-gpus 8 \
  --model.seq-len 16384 --data.seq-len 16384 --data.batch-size 16 --data.micro-batch-size 1 \
  --model.attn flash_attention_3 --model.compile --model.moe-router-dtype bfloat16 --model.ep "${EP_A2A_EP:-8}" \
  --model.offload none --model.ac-offloading None --max-steps "$STEPS" "$@" > "$OUT/sft.log" 2>&1
echo "exit=$? wall_s=$(( $(date +%s) - start ))" >> "$OUT/sft.log"
