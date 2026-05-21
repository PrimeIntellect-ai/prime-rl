#!/bin/bash
# Pod run script: takes a list of config files and runs them sequentially.
# Usage: ./pod-run.sh config1.toml config2.toml ...
# Configs are paths relative to the prime-rl repo root.

set -o pipefail

REPO_DIR="$HOME/prime-rl"
LOG_DIR="$REPO_DIR/configs/sweep-v2/logs"
mkdir -p "$LOG_DIR"

# Source env vars
export WANDB_API_KEY="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"
export PRIME_API_KEY="pit_f920c4b79c9c5568dda20eced3de43683dfdb6faea1e64063da80bad0cc0e781"
export PRIME_TEAM_ID="cmlr3u2er002zhr01tj8f48ts"
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config1.toml> [config2.toml] ..."
    echo "Configs are relative to $REPO_DIR"
    exit 1
fi

TOTAL=$#
PASSED=0
FAILED=0
FAILED_NAMES=""

TEACHER_PID=""

start_teacher() {
    if [ -n "$TEACHER_PID" ] && kill -0 "$TEACHER_PID" 2>/dev/null; then
        echo "Teacher already running (PID=$TEACHER_PID)"
        return
    fi
    echo "Starting teacher server on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 uv run inference \
        --model.name Qwen/Qwen3-0.6B \
        --server.port 8001 \
        --gpu-memory-utilization 0.5 \
        --model.enforce-eager &
    TEACHER_PID=$!
    echo "Waiting for teacher to be ready..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q "data"; then
            echo "Teacher ready (PID=$TEACHER_PID)"
            return
        fi
        sleep 5
    done
    echo "ERROR: Teacher failed to start within 10 minutes"
    kill "$TEACHER_PID" 2>/dev/null || true
    TEACHER_PID=""
    return 1
}

stop_teacher() {
    if [ -n "$TEACHER_PID" ]; then
        echo "Stopping teacher server (PID=$TEACHER_PID)..."
        kill "$TEACHER_PID" 2>/dev/null || true
        wait "$TEACHER_PID" 2>/dev/null || true
        TEACHER_PID=""
    fi
}

needs_teacher() {
    local config="$1"
    grep -q 'training_mode = "opd"' "$config" 2>/dev/null
}

run_config() {
    local config="$1"
    local name
    name=$(basename "$config" .toml)
    local logfile="$LOG_DIR/${name}.log"

    echo "=========================================="
    echo "Starting: $name ($(date))"
    echo "Config:   $config"
    echo "Log:      $logfile"
    echo "=========================================="

    # Start teacher if needed for OPD configs
    if needs_teacher "$config"; then
        if ! start_teacher; then
            echo "FAIL: $name - teacher server failed to start"
            FAILED=$((FAILED + 1))
            FAILED_NAMES="$FAILED_NAMES $name"
            return
        fi
    fi

    if uv run rl @ "$config" --clean-output-dir 2>&1 | tee "$logfile"; then
        echo "OK: $name completed successfully ($(date))"
        PASSED=$((PASSED + 1))
    else
        echo "FAIL: $name failed with exit code $? ($(date))"
        FAILED=$((FAILED + 1))
        FAILED_NAMES="$FAILED_NAMES $name"
    fi
    echo ""
}

# Classify configs into phases: RL/SFT first (no teacher), then OPD (needs teacher)
RL_SFT_CONFIGS=()
OPD_CONFIGS=()

for config in "$@"; do
    # Resolve relative paths
    if [[ "$config" != /* ]]; then
        config="$REPO_DIR/$config"
    fi
    if needs_teacher "$config"; then
        OPD_CONFIGS+=("$config")
    else
        RL_SFT_CONFIGS+=("$config")
    fi
done

echo "=== Assigned configs ==="
echo "  RL/SFT (no teacher): ${#RL_SFT_CONFIGS[@]}"
echo "  OPD (needs teacher): ${#OPD_CONFIGS[@]}"
echo "  Total: $TOTAL"
echo ""

# Phase 1: RL and SFT configs
if [ ${#RL_SFT_CONFIGS[@]} -gt 0 ]; then
    echo "=== PHASE 1: RL/SFT runs ==="
    for config in "${RL_SFT_CONFIGS[@]}"; do
        run_config "$config"
    done
fi

# Phase 2: OPD configs (need teacher)
if [ ${#OPD_CONFIGS[@]} -gt 0 ]; then
    echo "=== PHASE 2: OPD runs (with teacher) ==="
    for config in "${OPD_CONFIGS[@]}"; do
        run_config "$config"
    done
    stop_teacher
fi

echo "=========================================="
echo "ALL RUNS COMPLETE ($(date))"
echo "  Passed: $PASSED / $TOTAL"
echo "  Failed: $FAILED / $TOTAL"
if [ -n "$FAILED_NAMES" ]; then
    echo "  Failed configs:$FAILED_NAMES"
fi
echo "=========================================="

echo ""
echo "=== RESULTS SUMMARY ==="
for log in "$LOG_DIR"/*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    evals=$(grep "Evaluated" "$log" | tail -1)
    echo "  $name: $evals"
done
