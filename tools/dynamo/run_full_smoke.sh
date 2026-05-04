#!/bin/bash
# Full Dynamo + prime-rl smoke test: orchestrator + trainer
# GPU 0: Dynamo inference (already running)
# GPU 1: prime-rl trainer (single GPU, uses torchrun)
#
# Directory structure:
#   /tmp/dynamo_smoke_outputs/             <- trainer output_dir
#   /tmp/dynamo_smoke_outputs/run_default/ <- orchestrator output_dir (run_* naming required)

BASE_DIR=/tmp/dynamo_smoke_outputs
ORCH_DIR=$BASE_DIR/run_default
mkdir -p "$ORCH_DIR"

cd /home/biswaranjanp/dev/rl/prime-rl

# Start orchestrator writing to run_default (no GPU needed - just API calls to Dynamo)
echo "[smoke] Starting orchestrator (output: $ORCH_DIR)..."
BENCH_API_KEY=EMPTY CUDA_VISIBLE_DEVICES="" \
  uv run orchestrator \
    @ /tmp/dynamo_smoke_rl.toml \
    --output-dir "$ORCH_DIR" \
    --max-concurrent 4 \
  > /tmp/smoke_orchestrator.log 2>&1 &
ORCH_PID=$!
echo "[smoke] Orchestrator PID: $ORCH_PID"

# Give orchestrator time to write first batch
sleep 12

# Start trainer on GPU 1 via uv run torchrun (uses project venv Python, needed for torch.distributed)
echo "[smoke] Starting trainer (output: $BASE_DIR)..."
CUDA_VISIBLE_DEVICES=1 uv run torchrun \
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
