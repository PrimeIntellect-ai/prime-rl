#!/bin/bash
# No set -e — we want all runs to attempt even if one fails

export WANDB_API_KEY="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"
export PRIME_API_KEY="pit_f920c4b79c9c5568dda20eced3de43683dfdb6faea1e64063da80bad0cc0e781"
export PRIME_TEAM_ID="cmlr3u2er002zhr01tj8f48ts"

SWEEP_DIR="configs/sweep-v2"
LOG_DIR="configs/sweep-v2/logs"
mkdir -p "$LOG_DIR"

run_config() {
    local config="$1"
    local name=$(basename "$config" .toml)
    echo "=========================================="
    echo "Starting: $name ($(date))"
    echo "=========================================="
    if uv run rl @ "$config" --clean-output-dir 2>&1 | tee "$LOG_DIR/${name}.log"; then
        echo "OK: $name completed successfully ($(date))"
    else
        echo "FAIL: $name failed with exit code $? ($(date))"
    fi
    echo ""
}

echo "=== PHASE 1: RL runs (no teacher needed) ==="
for env in reverse-text gsm8k wordle alphabet-sort; do
    run_config "$SWEEP_DIR/${env}-rl.toml"
done

echo "=== PHASE 2: SFT runs (PI inference teacher) ==="
for env in reverse-text gsm8k wordle alphabet-sort; do
    run_config "$SWEEP_DIR/${env}-sft.toml"
done

echo "=== PHASE 3: OPD runs (local vLLM teacher) ==="
echo "Starting teacher server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 uv run inference \
    --model.name Qwen/Qwen3-0.6B \
    --server.port 8001 \
    --gpu-memory-utilization 0.5 \
    --model.enforce-eager &
TEACHER_PID=$!

echo "Waiting for teacher to be ready..."
until curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q "data"; do
    sleep 5
done
echo "Teacher ready (PID=$TEACHER_PID)"

for env in reverse-text gsm8k wordle alphabet-sort; do
    run_config "$SWEEP_DIR/${env}-opd.toml"
done

echo "Killing teacher server..."
kill $TEACHER_PID 2>/dev/null || true
wait $TEACHER_PID 2>/dev/null || true

echo "=========================================="
echo "SWEEP COMPLETE ($(date))"
echo "=========================================="

echo ""
echo "=== RESULTS SUMMARY ==="
for log in "$LOG_DIR"/*.log; do
    name=$(basename "$log" .log)
    evals=$(grep "Evaluated" "$log" | tail -1)
    echo "$name: $evals"
done
