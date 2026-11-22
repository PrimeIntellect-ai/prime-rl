#!/usr/bin/env bash
# Azure OpenAI eval using the same GlyphBench/verifiers settings as vLLM evals.
#
# Required:
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_RESPONSES_ENDPOINT
# Optional:
#   AZURE_OPENAI_MODEL

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

if [ -z "${AZURE_OPENAI_API_KEY:-}" ] || [ -z "${AZURE_OPENAI_RESPONSES_ENDPOINT:-}" ]; then
    echo "AZURE_OPENAI_API_KEY and AZURE_OPENAI_RESPONSES_ENDPOINT are required" >&2
    exit 2
fi

TASK_IDS=${TASK_IDS:-'["glyphbench/minigrid-empty-5x5-v0"]'}
extras=(--extra eval-client --extra tracking)
probe_modules=(prime_cli wandb)
case "$TASK_IDS" in
    *craftaxfull*|null) extras+=(--extra craftax); probe_modules+=(craftax) ;;
esac
case "$TASK_IDS" in
    *nethack*|null) extras+=(--extra nethack); probe_modules+=(nle) ;;
esac
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.cache/glyphbench-eval-all}"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] \
    || ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c \
        'import importlib, sys; [importlib.import_module(name) for name in sys.argv[1:]]' \
        "${probe_modules[@]}" >/dev/null 2>&1 \
    || [ "${GLYPHBENCH_AZURE_SYNC:-0}" = "1" ]; then
    uv sync --frozen "${extras[@]}"
fi

# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
eval "$(python "$REPO_ROOT/src/glyphbench/eval_defaults.py" shell)"

AZURE_OPENAI_RESPONSES_ENDPOINT="${AZURE_OPENAI_RESPONSES_ENDPOINT:-}"
AZURE_OPENAI_MODEL="${AZURE_OPENAI_MODEL:-gpt-5.4}"
MODEL="${MODEL:-$AZURE_OPENAI_MODEL}"
AZURE_PROXY_HOST="${AZURE_PROXY_HOST:-127.0.0.1}"
if [ -z "${AZURE_PROXY_PORT:-}" ]; then
    AZURE_PROXY_PORT=8010
fi
BASE_URL="http://${AZURE_PROXY_HOST}:${AZURE_PROXY_PORT}/v1"
AZURE_PROXY_LOG="${AZURE_PROXY_LOG:-$REPO_ROOT/outputs/eval/azure-proxy-${AZURE_PROXY_PORT}.log}"

API_KEY_VAR=${API_KEY_VAR:-AZURE_PROXY_API_KEY}
export "$API_KEY_VAR=${!API_KEY_VAR:-EMPTY}"
export PRIME_API_KEY=${PRIME_API_KEY:-EMPTY}

NUM_EPISODES=${NUM_EPISODES:-1}
ROLLOUTS_PER_EXAMPLE=${ROLLOUTS_PER_EXAMPLE:-$GLYPHBENCH_EVAL_ROLLOUTS_PER_EXAMPLE}
SEED=${SEED:-$GLYPHBENCH_EVAL_SEED}
N_FRAMES=${N_FRAMES:-$GLYPHBENCH_EVAL_N_FRAMES}
MAX_TURNS=${MAX_TURNS:-}
HARNESS=${HARNESS:-}
MAX_TOKENS=${MAX_TOKENS-$GLYPHBENCH_EVAL_MAX_TOKENS}
MEMORY_UPDATE_MAX_TOKENS=${MEMORY_UPDATE_MAX_TOKENS-$GLYPHBENCH_EVAL_MEMORY_UPDATE_MAX_TOKENS}
TEMPERATURE=${TEMPERATURE:-$GLYPHBENCH_EVAL_TEMPERATURE}
SAMPLING_ARGS=${SAMPLING_ARGS:-$GLYPHBENCH_EVAL_SAMPLING_ARGS}
MAX_CONCURRENT=${MAX_CONCURRENT:-}
SKIP_UPLOAD=${SKIP_UPLOAD:-1}
SAVE_RESULTS=${SAVE_RESULTS:-1}
SAFE_MODEL="$(printf '%s' "$AZURE_OPENAI_MODEL" | tr -cs 'A-Za-z0-9_.-' '_')"
EVAL_RUN_NAME=${EVAL_RUN_NAME:-"azure-${SAFE_MODEL}-normal-$(date -u +%Y%m%dT%H%M%SZ)"}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-$REPO_ROOT/outputs/eval/$EVAL_RUN_NAME}
SAVE_GIF=${SAVE_GIF:-1}
GIF_OUTPUT_DIR=${GIF_OUTPUT_DIR:-$EVAL_OUTPUT_DIR/gifs}
GIF_FPS=${GIF_FPS:-8}
GIF_MAX_FRAMES=${GIF_MAX_FRAMES:-4000}
WANDB_LOG=${WANDB_LOG:-0}
WANDB_REQUIRED=${WANDB_REQUIRED:-1}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-$EVAL_RUN_NAME}
WANDB_RUN_ID=${WANDB_RUN_ID:-$WANDB_RUN_NAME}
WANDB_TASK=${WANDB_TASK:-}
WANDB_REASONING_EFFORT=${WANDB_REASONING_EFFORT:-${AZURE_OPENAI_REASONING_EFFORT:-default}}

mkdir -p "$(dirname "$AZURE_PROXY_LOG")"
: > "$AZURE_PROXY_LOG"
AZURE_OPENAI_RESPONSES_ENDPOINT="$AZURE_OPENAI_RESPONSES_ENDPOINT" \
AZURE_OPENAI_MODEL="$AZURE_OPENAI_MODEL" \
AZURE_OPENAI_API_KEY="$AZURE_OPENAI_API_KEY" \
AZURE_OPENAI_TIMEOUT="${AZURE_OPENAI_TIMEOUT:-600}" \
AZURE_OPENAI_INVALID_PROMPT_RETRIES="${AZURE_OPENAI_INVALID_PROMPT_RETRIES:-2}" \
AZURE_OPENAI_CONTENT_FILTER_RETRIES="${AZURE_OPENAI_CONTENT_FILTER_RETRIES:-2}" \
AZURE_OPENAI_MODEL_ERROR_RETRIES="${AZURE_OPENAI_MODEL_ERROR_RETRIES:-4}" \
AZURE_OPENAI_TRANSIENT_RETRIES="${AZURE_OPENAI_TRANSIENT_RETRIES:-4}" \
"$UV_PROJECT_ENVIRONMENT/bin/python" "$REPO_ROOT/scripts/azure_responses_proxy.py" \
    --host "$AZURE_PROXY_HOST" \
    --port "$AZURE_PROXY_PORT" \
    > "$AZURE_PROXY_LOG" 2>&1 &
AZURE_PROXY_PID=$!
trap 'kill "$AZURE_PROXY_PID" 2>/dev/null || true' EXIT

echo ">>> Azure proxy pid=$AZURE_PROXY_PID base_url=$BASE_URL log=$AZURE_PROXY_LOG"
deadline=$((SECONDS + 60))
until curl -fs --max-time 3 "$BASE_URL/models" >/dev/null 2>&1; do
    if ! kill -0 "$AZURE_PROXY_PID" 2>/dev/null; then
        echo "Azure proxy exited before becoming ready" >&2
        tail -80 "$AZURE_PROXY_LOG" >&2 || true
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "Azure proxy did not become ready" >&2
        tail -80 "$AZURE_PROXY_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

MAX_TURNS_ARG=""
if [ -n "$MAX_TURNS" ]; then
    MAX_TURNS_ARG=", \"max_turns\": $MAX_TURNS"
fi
HARNESS_ARG=""
if [ -n "$HARNESS" ]; then
    HARNESS_ARG=", \"harness\": \"$HARNESS\""
fi
MEMORY_UPDATE_MAX_TOKENS_ARG=""
if [ "$MEMORY_UPDATE_MAX_TOKENS" = "none" ] || [ "$MEMORY_UPDATE_MAX_TOKENS" = "native" ]; then
    MEMORY_UPDATE_MAX_TOKENS_ARG=', "memory_update_max_tokens": null'
elif [ -n "$MEMORY_UPDATE_MAX_TOKENS" ]; then
    MEMORY_UPDATE_MAX_TOKENS_ARG=", \"memory_update_max_tokens\": $MEMORY_UPDATE_MAX_TOKENS"
fi
MAX_OUTPUT_TOKENS_JSON="$MAX_TOKENS"
if [ "$MAX_TOKENS" = "none" ] || [ "$MAX_TOKENS" = "native" ] || [ -z "$MAX_TOKENS" ]; then
    MAX_OUTPUT_TOKENS_JSON=null
fi
SAVE_GIF_JSON=false
if [ "$SAVE_GIF" = "1" ]; then
    SAVE_GIF_JSON=true
fi

# Evaluate every task/seed row; num_episodes already controls seeds per task.
prime_args=(
    eval run glyphbench
    --api-client-type openai_chat_completions
    -m "$MODEL"
    -b "$BASE_URL"
    -k "$API_KEY_VAR"
    -n -1
    --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE"
    --sampling-args "$SAMPLING_ARGS"
    -a "{\"task_id\": $TASK_IDS, \"num_episodes\": $NUM_EPISODES, \"n_frames\": $N_FRAMES, \"seed\": $SEED, \"max_output_tokens\": $MAX_OUTPUT_TOKENS_JSON, \"save_gif\": $SAVE_GIF_JSON, \"gif_output_dir\": \"$GIF_OUTPUT_DIR\", \"gif_fps\": $GIF_FPS, \"gif_max_frames\": $GIF_MAX_FRAMES$MAX_TURNS_ARG$HARNESS_ARG$MEMORY_UPDATE_MAX_TOKENS_ARG}"
)

if [ "$MAX_OUTPUT_TOKENS_JSON" != "null" ]; then
    prime_args+=(--max-tokens "$MAX_TOKENS")
fi
if [ "$TEMPERATURE" != "native" ] && [ "$TEMPERATURE" != "none" ]; then
    prime_args+=(--temperature "$TEMPERATURE")
fi

if [ -n "$MAX_CONCURRENT" ]; then
    prime_args+=(--max-concurrent "$MAX_CONCURRENT")
fi
if [ "$SKIP_UPLOAD" = "1" ]; then
    prime_args+=(--skip-upload)
fi
if [ "$SAVE_RESULTS" = "1" ]; then
    prime_args+=(--save-results)
fi
if [ -n "$EVAL_OUTPUT_DIR" ]; then
    prime_args+=(--output-dir "$EVAL_OUTPUT_DIR")
fi

echo ">>> Running Azure eval against $AZURE_OPENAI_MODEL via $BASE_URL"
eval_status=0
prime "${prime_args[@]}" || eval_status=$?
mkdir -p "$EVAL_OUTPUT_DIR"
curl -fsS --max-time 10 "$BASE_URL/proxy-metrics" \
    > "$EVAL_OUTPUT_DIR/azure_proxy_metrics.json" || true

wandb_status=0
if [ "$WANDB_LOG" = "1" ]; then
    tracking_args=()
    if [ -n "$WANDB_TASK" ]; then
        tracking_args+=(--task "$WANDB_TASK")
    fi
    python -m glyphbench.eval_tracking \
        --results-root "$EVAL_OUTPUT_DIR" \
        --harness normal \
        --model "$AZURE_OPENAI_MODEL" \
        --reasoning-effort "$WANDB_REASONING_EFFORT" \
        --run-name "$WANDB_RUN_NAME" \
        --run-id "$WANDB_RUN_ID" \
        "${tracking_args[@]}" || wandb_status=$?
fi

if [ "$eval_status" -ne 0 ]; then
    exit "$eval_status"
fi
if [ "$WANDB_REQUIRED" = "1" ] && [ "$wandb_status" -ne 0 ]; then
    exit "$wandb_status"
fi
