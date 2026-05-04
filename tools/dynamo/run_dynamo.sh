#!/bin/bash
# Launch Dynamo frontend + vLLM worker for smoke testing with prime-rl.
#
# Prerequisites:
#   - Dynamo virtualenv at $DYNAMO_VENV (default: ~/dev/dynamo/dynamo)
#   - etcd + NATS running (dynamo depends on them)
#
# Usage:
#   ./tools/dynamo/run_dynamo.sh
#   CUDA_VISIBLE_DEVICES=0 ./tools/dynamo/run_dynamo.sh
#   DYNAMO_MODEL=my-org/my-model ./tools/dynamo/run_dynamo.sh

set -euo pipefail

DYNAMO_VENV="${DYNAMO_VENV:-$HOME/dev/dynamo/dynamo}"
DYNAMO_MODEL="${DYNAMO_MODEL:-PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${LOG_DIR:-/tmp}"

export CUDA_VISIBLE_DEVICES

source "$DYNAMO_VENV/bin/activate"

echo "[dynamo-smoke] Starting frontend..."
python -m dynamo.frontend > "$LOG_DIR/dynamo_frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "[dynamo-smoke] Frontend PID: $FRONTEND_PID (log: $LOG_DIR/dynamo_frontend.log)"

sleep 5

echo "[dynamo-smoke] Starting vLLM worker (model: $DYNAMO_MODEL)..."
# --gpu-memory-utilization=0.45 caps vLLM at ~45% of GPU memory so the
# colocated trainer (run via tools/dynamo/run_full_smoke.sh) has room.
# On a single 96 GB GPU: ~43 GB to inference, ~50 GB to trainer.
# Override with GPU_MEM_UTIL=<float> if you have headroom or constraints.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm \
  --model "$DYNAMO_MODEL" \
  --enforce-eager \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  > "$LOG_DIR/dynamo_vllm.log" 2>&1 &
WORKER_PID=$!
echo "[dynamo-smoke] Worker PID: $WORKER_PID (log: $LOG_DIR/dynamo_vllm.log)"

echo "[dynamo-smoke] Both processes launched."
echo "[dynamo-smoke] Frontend log: $LOG_DIR/dynamo_frontend.log"
echo "[dynamo-smoke] Worker log:   $LOG_DIR/dynamo_vllm.log"
wait
