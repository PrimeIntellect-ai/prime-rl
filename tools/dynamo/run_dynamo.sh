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

DYNAMO_VENV="${DYNAMO_VENV:-/home/biswaranjanp/dev/rl/dynamo/.venv}"
DYNAMO_MODEL="${DYNAMO_MODEL:-PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT}"
DYNAMO_GPU_MEMORY_UTILIZATION="${DYNAMO_GPU_MEMORY_UTILIZATION:-0.50}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${LOG_DIR:-/tmp}"
PRIME_RL_SRC="${PRIME_RL_SRC:-/home/biswaranjanp/dev/rl/prime-rl/src}"

export CUDA_VISIBLE_DEVICES

source "$DYNAMO_VENV/bin/activate"

echo "[dynamo-smoke] Starting frontend..."
DYN_ENABLE_RL=true \
DYN_RL_PORT="${DYN_RL_PORT:-8001}" \
python -m dynamo.frontend > "$LOG_DIR/dynamo_frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "[dynamo-smoke] Frontend PID: $FRONTEND_PID (log: $LOG_DIR/dynamo_frontend.log)"

sleep 5

echo "[dynamo-smoke] Starting vLLM worker (model: $DYNAMO_MODEL)..."
PYTHONPATH="${PRIME_RL_SRC}${PYTHONPATH:+:$PYTHONPATH}" \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
python -m dynamo.vllm \
  --model "$DYNAMO_MODEL" \
  --enforce-eager \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --gpu-memory-utilization "$DYNAMO_GPU_MEMORY_UTILIZATION" \
  --enable-rl \
  --worker-extension-cls prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker \
  > "$LOG_DIR/dynamo_vllm.log" 2>&1 &
WORKER_PID=$!
echo "[dynamo-smoke] Worker PID: $WORKER_PID (log: $LOG_DIR/dynamo_vllm.log)"

echo "[dynamo-smoke] Both processes launched."
echo "[dynamo-smoke] Frontend log: $LOG_DIR/dynamo_frontend.log"
echo "[dynamo-smoke] Worker log:   $LOG_DIR/dynamo_vllm.log"
wait
