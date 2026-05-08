#!/bin/bash
# Full Dynamo + prime-rl smoke test: orchestrator + trainer (single-GPU colocated).
#
# Layout (this machine has 1× NVIDIA RTX PRO 6000 Blackwell, 96 GB):
#   GPU 0: Dynamo inference (already running, started separately) AND trainer
#   no GPU: orchestrator (HTTP-only)
#
# Prerequisite — Dynamo must already be running on GPU 0 with bounded vLLM
# memory so the trainer has room to load FSDP-sharded weights + optimizer.
# Recommended:
#
#   CUDA_VISIBLE_DEVICES=0 ./tools/dynamo/run_dynamo.sh
#
# AND edit run_dynamo.sh's vllm command to add:
#
#   --gpu-memory-utilization 0.45
#
# That leaves ~50 GB free for the trainer. For a Qwen3-0.6B smoke test the
# trainer needs ~10 GB; 50 GB is comfortable.
#
# Directory structure:
#   /tmp/dynamo_smoke_outputs/             <- trainer output_dir
#   /tmp/dynamo_smoke_outputs/run_default/ <- orchestrator output_dir (run_* naming required)

set -euo pipefail

BASE_DIR=/tmp/dynamo_smoke_outputs
ORCH_DIR=$BASE_DIR/run_default
mkdir -p "$ORCH_DIR"

cd /home/biswaranjanp/dev/rl/prime-rl

# Sanity check: warn if GPU 0 isn't visible.
if ! nvidia-smi -i 0 --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  echo "[smoke] ERROR: GPU 0 not visible to nvidia-smi" >&2
  exit 1
fi

# Sanity check: warn if Dynamo isn't already up on the default frontend port.
if ! curl --silent --max-time 2 http://localhost:8000/health >/dev/null 2>&1; then
  echo "[smoke] ERROR: Dynamo frontend not reachable at http://localhost:8000/health" >&2
  echo "[smoke]        Start it first via:  ./tools/dynamo/run_dynamo.sh"  >&2
  exit 1
fi

# Start orchestrator writing to run_default (no GPU needed — just API calls to Dynamo).
echo "[smoke] Starting orchestrator (output: $ORCH_DIR)..."
BENCH_API_KEY=EMPTY CUDA_VISIBLE_DEVICES="" \
  uv run orchestrator \
    @ /tmp/dynamo_smoke_rl.toml \
    --output-dir "$ORCH_DIR" \
    --max-concurrent 4 \
  > /tmp/smoke_orchestrator.log 2>&1 &
ORCH_PID=$!
echo "[smoke] Orchestrator PID: $ORCH_PID"

# Give orchestrator time to write first batch.
sleep 12

# Start trainer on GPU 0 (colocated with Dynamo) via uv run torchrun.
# uv run handles the project venv Python, needed for torch.distributed bootstrap.
echo "[smoke] Starting trainer (output: $BASE_DIR, colocated on GPU 0)..."
CUDA_VISIBLE_DEVICES=0 uv run torchrun \
  --nproc-per-node=1 \
  --rdzv-endpoint=localhost:29510 \
  --rdzv-id=smoke_$(date +%s) \
  -m prime_rl.trainer.rl.train \
    @ /tmp/dynamo_smoke_trainer.toml \
    --output-dir "$BASE_DIR" \
  > /tmp/smoke_trainer.log 2>&1 &
TRAINER_PID=$!
echo "[smoke] Trainer PID: $TRAINER_PID"

echo "[smoke] Waiting for both processes..."
wait $ORCH_PID
ORCH_EXIT=$?
wait $TRAINER_PID
TRAINER_EXIT=$?

echo "[smoke] Orchestrator exit: $ORCH_EXIT"
echo "[smoke] Trainer exit: $TRAINER_EXIT"
