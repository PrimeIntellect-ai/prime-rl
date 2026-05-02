#!/bin/bash
# Full Dynamo + prime-rl smoke test: orchestrator + trainer.
#
# GPU 0: Dynamo inference (must already be running -- see run_dynamo.sh)
# GPU 1: prime-rl trainer (single GPU, uses torchrun)
#
# Directory structure created:
#   $OUTPUT_DIR/             <- trainer output_dir
#   $OUTPUT_DIR/run_default/ <- orchestrator output_dir (run_* naming required)
#
# Usage:
#   # Short run (5 steps, default):
#   ./tools/dynamo/run_smoke_test.sh
#
#   # Long run (20 steps):
#   ./tools/dynamo/run_smoke_test.sh --long
#
#   # Custom output dir:
#   OUTPUT_DIR=/tmp/my_run ./tools/dynamo/run_smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"

# Defaults
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dynamo_smoke_outputs}"
TRAINER_GPU="${TRAINER_GPU:-1}"
RDZV_PORT="${RDZV_PORT:-29510}"

# Pick short vs long config
if [[ "${1:-}" == "--long" ]]; then
    ORCH_CONFIG="$CONFIG_DIR/smoke_rl_long.toml"
    TRAINER_CONFIG="$CONFIG_DIR/smoke_trainer_long.toml"
    ORCH_LOG="${OUTPUT_DIR}/smoke_long_orchestrator.log"
    TRAINER_LOG="${OUTPUT_DIR}/smoke_long_trainer.log"
    echo "[smoke] Using LONG configs (20 steps)"
else
    ORCH_CONFIG="$CONFIG_DIR/smoke_rl.toml"
    TRAINER_CONFIG="$CONFIG_DIR/smoke_trainer.toml"
    ORCH_LOG="${OUTPUT_DIR}/smoke_orchestrator.log"
    TRAINER_LOG="${OUTPUT_DIR}/smoke_trainer.log"
    echo "[smoke] Using SHORT configs (5 steps)"
fi

ORCH_DIR="$OUTPUT_DIR/run_default"
mkdir -p "$ORCH_DIR"

cd "$REPO_DIR"

# Start orchestrator (no GPU needed -- just API calls to Dynamo)
echo "[smoke] Starting orchestrator (output: $ORCH_DIR)..."
BENCH_API_KEY=EMPTY CUDA_VISIBLE_DEVICES="" \
  uv run orchestrator \
    @ "$ORCH_CONFIG" \
    --output-dir "$ORCH_DIR" \
    --max-concurrent 4 \
  > "$ORCH_LOG" 2>&1 &
ORCH_PID=$!
echo "[smoke] Orchestrator PID: $ORCH_PID (log: $ORCH_LOG)"

# Give orchestrator time to write first batch
sleep 12

# Start trainer on GPU via uv run torchrun (uses project venv Python, needed for torch.distributed)
echo "[smoke] Starting trainer (output: $OUTPUT_DIR)..."
CUDA_VISIBLE_DEVICES="$TRAINER_GPU" uv run torchrun \
  --nproc-per-node=1 \
  --rdzv-endpoint="localhost:$RDZV_PORT" \
  --rdzv-id="smoke_$(date +%s)" \
  -m prime_rl.trainer.rl.train \
    @ "$TRAINER_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
  > "$TRAINER_LOG" 2>&1 &
TRAINER_PID=$!
echo "[smoke] Trainer PID: $TRAINER_PID (log: $TRAINER_LOG)"

echo "[smoke] Waiting for both processes..."

wait $ORCH_PID
ORCH_EXIT=$?
wait $TRAINER_PID
TRAINER_EXIT=$?

echo ""
echo "[smoke] Orchestrator exit: $ORCH_EXIT"
echo "[smoke] Trainer exit: $TRAINER_EXIT"

if [[ $ORCH_EXIT -eq 0 && $TRAINER_EXIT -eq 0 ]]; then
    echo "[smoke] Smoke test PASSED."
else
    echo "[smoke] Smoke test FAILED."
    exit 1
fi
