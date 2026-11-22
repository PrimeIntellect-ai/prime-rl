#!/usr/bin/env bash
# Standalone "Pro" harness for the long-horizon games (CraftaxFull / NetHack).
#
# Unlike run_full.sh / run_azure.sh (which go through `prime eval run` +
# verifiers), this drives the env directly via `python -m glyphbench.pro_harness`
# and makes its own LLM calls, so it owns content-filter retries, its grounded
# action/memory loop, and an uncapped reasoning budget.
#
# Swap the inference server with BACKEND (openai|azure); swap the game with
# TASK. Examples:
#
#   # vLLM / any OpenAI-compatible server
#   TASK=glyphbench/craftaxfull-v0 MODEL=Qwen/Qwen3.5-4B \
#     BASE_URL=http://localhost:8000/v1 bash eval/run_pro.sh
#
#   # Azure Responses API (GPT-5.x), content-filter robust
#   BACKEND=azure TASK=glyphbench/nethack-full-v0 \
#     AZURE_OPENAI_MODEL=gpt-5.4 REASONING_EFFORT=high bash eval/run_pro.sh
#
# Azure needs AZURE_OPENAI_API_KEY + AZURE_OPENAI_RESPONSES_ENDPOINT (read from
# .env if present).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

TASK="${TASK:-glyphbench/craftaxfull-v0}"
BACKEND="${BACKEND:-openai}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SEED="${SEED:-42}"
MAX_TURNS="${MAX_TURNS:-}"            # empty => env-native horizon
if [ -n "${REASONING_EFFORT+x}" ]; then
    REASONING_EFFORT="$REASONING_EFFORT"
else
    REASONING_EFFORT="${AZURE_OPENAI_REASONING_EFFORT:-}"
fi
if [ "$REASONING_EFFORT" = "default" ] || [ "$REASONING_EFFORT" = "native" ]; then
    REASONING_EFFORT=""
fi
# Empty => NO output cap (no reasoning truncation). Set an int to cap.
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"
TEMPERATURE="${TEMPERATURE:-native}"
TOP_P="${TOP_P:-native}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/pro_harness}"
RUN_TAG="${RUN_TAG:-}"
HARNESS_VARIANT="${HARNESS_VARIANT:-pro}"
REASONING_MODE="${REASONING_MODE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
VIDEO_FPS="${VIDEO_FPS:-8}"
MEMORY_UPDATE_EVERY="${MEMORY_UPDATE_EVERY:-1}"

# Pick the venv + extras the task needs. CraftaxFull pulls the craftax
# submodule (textures); NetHack needs the local NLE fork.
case "$TASK" in
    *nethack*) EXTRAS=(--extra nethack --extra tracking); VENV_DEFAULT="$HOME/.cache/glyphbench-pro-nethack" ;;
    *)         EXTRAS=(--extra craftax --extra tracking); VENV_DEFAULT="$HOME/.cache/glyphbench-pro-craftax" ;;
esac
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$VENV_DEFAULT}"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] || [ "${GLYPHBENCH_PRO_SYNC:-0}" = "1" ]; then
    uv sync --frozen "${EXTRAS[@]}"
fi
# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

args=(
    -m glyphbench.pro_harness
    --task "$TASK"
    --backend "$BACKEND"
    --harness-variant "$HARNESS_VARIANT"
    --num-episodes "$NUM_EPISODES"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --video-fps "$VIDEO_FPS"
    --memory-update-every "$MEMORY_UPDATE_EVERY"
)
[ -n "$MAX_TURNS" ] && args+=(--max-turns "$MAX_TURNS")
[ -n "$REASONING_EFFORT" ] && args+=(--reasoning-effort "$REASONING_EFFORT")
[ -n "$REASONING_MODE" ] && args+=(--reasoning-mode "$REASONING_MODE")
[ -n "$MAX_OUTPUT_TOKENS" ] && args+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")
[ "$TEMPERATURE" != "native" ] && [ "$TEMPERATURE" != "none" ] && args+=(--temperature "$TEMPERATURE")
[ "$TOP_P" != "native" ] && [ "$TOP_P" != "none" ] && args+=(--top-p "$TOP_P")
[ -n "$RUN_TAG" ] && args+=(--run-tag "$RUN_TAG")
[ -n "$WANDB_PROJECT" ] && args+=(--wandb-project "$WANDB_PROJECT")
[ -n "$WANDB_GROUP" ] && args+=(--wandb-group "$WANDB_GROUP")
[ -n "$WANDB_NAME" ] && args+=(--wandb-name "$WANDB_NAME")
[ -n "$WANDB_RUN_ID" ] && args+=(--wandb-run-id "$WANDB_RUN_ID")

if [ "$BACKEND" = "azure" ]; then
    args+=(--azure-model "${AZURE_OPENAI_MODEL:-gpt-5.4}")
    [ -n "${AZURE_OPENAI_RESPONSES_ENDPOINT:-}" ] && args+=(--azure-endpoint "$AZURE_OPENAI_RESPONSES_ENDPOINT")
    [ -n "${AZURE_OPENAI_POLICY_ID:-}" ] && args+=(--azure-policy-id "$AZURE_OPENAI_POLICY_ID")
else
    args+=(--model "${MODEL:-Qwen/Qwen3.5-4B}")
    args+=(--base-url "${BASE_URL:-${VLLM_BASE_URL:-http://localhost:8000/v1}}")
    args+=(--api-key-env "${API_KEY_VAR:-OPENAI_API_KEY_LOCAL}")
fi

echo ">>> harness: task=$TASK backend=$BACKEND variant=$HARNESS_VARIANT episodes=$NUM_EPISODES"
exec python "${args[@]}"
