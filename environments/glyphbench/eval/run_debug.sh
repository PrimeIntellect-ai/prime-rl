#!/usr/bin/env bash
# Smoke eval via `prime eval run` against a local vLLM server.
#
# This script will:
#   1. Pre-check vLLM is reachable at $VLLM_BASE_URL.
#   2. If not, optionally start one in the background (set AUTO_START_VLLM=1).
#   3. Run a 1-env × 2-episode smoke eval through `prime eval run`.
#
# Usage:
#   bash eval/run_debug.sh                           # vLLM must already be up
#   AUTO_START_VLLM=1 bash eval/run_debug.sh         # start vLLM if needed (~60s)
#   MODEL=Qwen/Qwen3-1.7B bash eval/run_debug.sh
#   VLLM_BASE_URL=http://other-host:8000/v1 bash eval/run_debug.sh
#
# After it finishes, view results with: prime eval tui

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the project venv so `prime`, `vllm`, and `glyphbench` are on PATH/PYTHONPATH.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$REPO_ROOT/.venv" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Pull HF_TOKEN etc. from .env if present (for vLLM model downloads).
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi
eval "$(python "$REPO_ROOT/src/glyphbench/eval_defaults.py" shell)"

AUTO_START_VLLM=${AUTO_START_VLLM:-0}
MODEL=${MODEL:-$GLYPHBENCH_EVAL_MODEL}
if [ -n "${VLLM_BASE_URL:-}" ]; then
    BASE_URL="$VLLM_BASE_URL"
else
    if [ "$AUTO_START_VLLM" = "1" ]; then
        VLLM_PORT="${VLLM_PORT:-$((18000 + ($$ % 20000)))}"
    else
        VLLM_PORT="${VLLM_PORT:-8000}"
    fi
    BASE_URL="http://localhost:${VLLM_PORT}/v1"
fi
API_KEY_VAR=${API_KEY_VAR:-OPENAI_API_KEY_LOCAL}
if [ -z "${VLLM_LOG:-}" ]; then
    VLLM_LOG=/tmp/glyphbench-vllm.log
fi
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}
VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-1200}
VLLM_POLL_INTERVAL=${VLLM_POLL_INTERVAL:-5}
SKIP_UPLOAD=${SKIP_UPLOAD:-1}                    # local/debug runs do not require prime login by default
SAVE_RESULTS=${SAVE_RESULTS:-1}                  # keep results.jsonl for gb replay by default
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-}              # optional prime eval --output-dir base

# Eval knobs (override via env)
N_FRAMES=${N_FRAMES:-$GLYPHBENCH_EVAL_N_FRAMES} # frame-stack history window (0 = stateless per turn)
NUM_EPISODES=${NUM_EPISODES:-2}                 # distinct (env, seed) rows per env
ROLLOUTS_PER_EXAMPLE=${ROLLOUTS_PER_EXAMPLE:-$GLYPHBENCH_EVAL_ROLLOUTS_PER_EXAMPLE} # repeats per row
SEED=${SEED:-$GLYPHBENCH_EVAL_SEED}
TASK_IDS=${TASK_IDS:-'["glyphbench/minigrid-empty-5x5-v0"]'}
MAX_TURNS=${MAX_TURNS:-}                        # optional per-episode cap for smoke tests
HARNESS=${HARNESS:-}                            # optional load_environment harness
MAX_TOKENS=${MAX_TOKENS:-$GLYPHBENCH_EVAL_MAX_TOKENS}
MEMORY_UPDATE_MAX_TOKENS=${MEMORY_UPDATE_MAX_TOKENS:-$GLYPHBENCH_EVAL_MEMORY_UPDATE_MAX_TOKENS}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$GLYPHBENCH_EVAL_MAX_MODEL_LEN}

# Sampling: clean default train/eval profile from glyphbench.eval_defaults.
TEMPERATURE=${TEMPERATURE:-$GLYPHBENCH_EVAL_TEMPERATURE}
SAMPLING_ARGS=${SAMPLING_ARGS:-$GLYPHBENCH_EVAL_SAMPLING_ARGS}

# vLLM doesn't validate the key — set a placeholder if not already set.
export "$API_KEY_VAR=${!API_KEY_VAR:-EMPTY}"
# prime CLI v0.5.x still checks for its platform key even when --skip-upload is
# set; a placeholder is enough for fully local evals.
export PRIME_API_KEY=${PRIME_API_KEY:-EMPTY}

# ---------------------------------------------------------------------------
# vLLM reachability check
# ---------------------------------------------------------------------------

is_vllm_up() {
    curl -fs --max-time 3 "${BASE_URL%/v1}/v1/models" > /dev/null 2>&1
}

dump_vllm_diagnostics() {
    local port="$1"
    echo "!!! vLLM diagnostics" >&2
    echo "    base_url=$BASE_URL" >&2
    echo "    log=$VLLM_LOG" >&2
    if [ -n "${VLLM_PID:-}" ]; then
        ps -o pid,ppid,stat,etime,pcpu,pmem,args -p "$VLLM_PID" >&2 || true
        pgrep -P "$VLLM_PID" | xargs -r ps -o pid,ppid,stat,etime,pcpu,pmem,args -p >&2 || true
    fi
    ls -lh "$VLLM_LOG" >&2 || true
    stat "$VLLM_LOG" >&2 || true
    if command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null | grep -E "(:${port} )" >&2 || true
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi >&2 || true
    fi
    echo "!!! Last 80 lines of $VLLM_LOG:" >&2
    tail -80 "$VLLM_LOG" >&2 || true
}

start_vllm_in_bg() {
    echo ">>> Starting vLLM: $MODEL on $BASE_URL  (logging to $VLLM_LOG)"
    local port
    port="$(echo "$BASE_URL" | sed -E 's|.*://[^:/]+:?([0-9]*)/?.*|\1|')"
    port="${port:-8000}"
    mkdir -p "$(dirname "$VLLM_LOG")"
    : > "$VLLM_LOG"
    local extra_args=()
    if [ -n "$VLLM_EXTRA_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_args=( $VLLM_EXTRA_ARGS )
    fi
    (
        export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
        export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
        # These defaults mitigate the pinned kernel for dense Qwen3.5.
        # Review docs/DEPENDENCIES.md before changing models or overriding them.
        exec vllm serve "$MODEL" --port "$port" --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --max-num-batched-tokens 8192 --enable-chunked-prefill \
            --compilation-config '{"custom_ops":["-silu_and_mul"]}' "${extra_args[@]}"
    ) > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "    vllm pid=$VLLM_PID"
    echo ">>> Waiting for vLLM to become ready (up to ${VLLM_STARTUP_TIMEOUT}s)..."
    local deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
    while [ $SECONDS -lt $deadline ]; do
        if is_vllm_up; then
            echo ">>> vLLM ready"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "!!! vLLM process died before becoming ready." >&2
            dump_vllm_diagnostics "$port"
            return 1
        fi
        sleep "$VLLM_POLL_INTERVAL"
    done
    echo "!!! vLLM didn't become ready within ${VLLM_STARTUP_TIMEOUT}s." >&2
    dump_vllm_diagnostics "$port"
    return 1
}

if ! is_vllm_up; then
    if [ "$AUTO_START_VLLM" = "1" ]; then
        # Also clean up when startup times out before the server becomes ready.
        trap '[ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true' EXIT
        start_vllm_in_bg || exit 1
    else
        cat >&2 <<EOF
!!! No vLLM server reachable at $BASE_URL.

Start one in another terminal:
    cd $REPO_ROOT
    source .venv/bin/activate
    set -a; source .env; set +a       # for HF_TOKEN
    vllm serve $MODEL --port 8000 --max-model-len $VLLM_MAX_MODEL_LEN --enforce-eager \\
        --max-num-batched-tokens 8192 --enable-chunked-prefill \\
        --compilation-config '{"custom_ops":["-silu_and_mul"]}' $VLLM_EXTRA_ARGS

… then re-run this script.

Alternatively, let this script auto-start it:
    AUTO_START_VLLM=1 bash eval/run_debug.sh
EOF
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

echo ">>> Running prime eval against $BASE_URL ($MODEL)"
MAX_TURNS_ARG=""
if [ -n "$MAX_TURNS" ]; then
  MAX_TURNS_ARG=", \"max_turns\": $MAX_TURNS"
fi
HARNESS_ARG=""
if [ -n "$HARNESS" ]; then
  HARNESS_ARG=", \"harness\": \"$HARNESS\""
fi
MEMORY_UPDATE_MAX_TOKENS_ARG=""
if [ -n "$MEMORY_UPDATE_MAX_TOKENS" ]; then
  MEMORY_UPDATE_MAX_TOKENS_ARG=", \"memory_update_max_tokens\": $MEMORY_UPDATE_MAX_TOKENS"
fi
SKIP_UPLOAD_ARG=""
if [ "$SKIP_UPLOAD" = "1" ]; then
  SKIP_UPLOAD_ARG="--skip-upload"
fi
SAVE_RESULTS_ARG=""
if [ "$SAVE_RESULTS" = "1" ]; then
  SAVE_RESULTS_ARG="--save-results"
fi
OUTPUT_DIR_ARGS=()
if [ -n "$EVAL_OUTPUT_DIR" ]; then
  OUTPUT_DIR_ARGS=(--output-dir "$EVAL_OUTPUT_DIR")
fi
# Evaluate every task/seed row; num_episodes already controls seeds per task.
prime_args=(
  eval run glyphbench
  --provider vllm
  -m "$MODEL"
  -b "$BASE_URL"
  -k "$API_KEY_VAR"
  -n -1
  --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --sampling-args "$SAMPLING_ARGS"
  "${OUTPUT_DIR_ARGS[@]}"
  -a "{\"task_id\": $TASK_IDS, \"num_episodes\": $NUM_EPISODES, \"n_frames\": $N_FRAMES, \"seed\": $SEED, \"max_output_tokens\": $MAX_TOKENS$MAX_TURNS_ARG$HARNESS_ARG$MEMORY_UPDATE_MAX_TOKENS_ARG}"
)
if [ -n "$SKIP_UPLOAD_ARG" ]; then
  prime_args+=("$SKIP_UPLOAD_ARG")
fi
if [ -n "$SAVE_RESULTS_ARG" ]; then
  prime_args+=("$SAVE_RESULTS_ARG")
fi
prime "${prime_args[@]}"
