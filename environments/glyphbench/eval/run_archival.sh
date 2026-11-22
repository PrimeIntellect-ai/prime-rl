#!/usr/bin/env bash
# Archival eval: runs the full-horizon atari originals + craftaxfull + nethack. Slow.
#
# Prereq: a vLLM server at $VLLM_BASE_URL serving the target model.
#
# Use this only when you specifically want to measure the open-ended /
# long-horizon performance ceiling. Default eval excludes these suites
# (see eval/run_full.sh).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi
eval "$(python "$REPO_ROOT/src/glyphbench/eval_defaults.py" shell)"

MODEL=${MODEL:-$GLYPHBENCH_EVAL_MODEL}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}
API_KEY_VAR=${API_KEY_VAR:-OPENAI_API_KEY_LOCAL}
EPISODES=${EPISODES:-5}
ROLLOUTS_PER_EXAMPLE=${ROLLOUTS_PER_EXAMPLE:-$GLYPHBENCH_EVAL_ROLLOUTS_PER_EXAMPLE}
SEED=${SEED:-$GLYPHBENCH_EVAL_SEED}
N_FRAMES=${N_FRAMES:-$GLYPHBENCH_EVAL_N_FRAMES}
MAX_TOKENS=${MAX_TOKENS:-$GLYPHBENCH_EVAL_MAX_TOKENS}
MEMORY_UPDATE_MAX_TOKENS=${MEMORY_UPDATE_MAX_TOKENS:-$GLYPHBENCH_EVAL_MEMORY_UPDATE_MAX_TOKENS}
TEMPERATURE=${TEMPERATURE:-$GLYPHBENCH_EVAL_TEMPERATURE}
SAMPLING_ARGS=${SAMPLING_ARGS:-$GLYPHBENCH_EVAL_SAMPLING_ARGS}

export "$API_KEY_VAR=${!API_KEY_VAR:-EMPTY}"
result_args=()
if [ "${SKIP_UPLOAD:-1}" = "1" ]; then
    result_args+=(--skip-upload)
    export PRIME_API_KEY=${PRIME_API_KEY:-EMPTY}
fi
if [ "${SAVE_RESULTS:-1}" = "1" ]; then
    result_args+=(--save-results)
fi
if [ -n "${EVAL_OUTPUT_DIR:-}" ]; then
    result_args+=(--output-dir "$EVAL_OUTPUT_DIR")
fi

# Evaluate every task/seed row; num_episodes already controls seeds per task.
prime eval run glyphbench "${result_args[@]}" \
  --provider vllm \
  -m "$MODEL" \
  -b "$BASE_URL" \
  -k "$API_KEY_VAR" \
  -n -1 \
  --rollouts-per-example "$ROLLOUTS_PER_EXAMPLE" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --sampling-args "$SAMPLING_ARGS" \
  -a "{\"num_episodes\": $EPISODES, \"n_frames\": $N_FRAMES, \"seed\": $SEED, \"max_output_tokens\": $MAX_TOKENS, \"memory_update_max_tokens\": $MEMORY_UPDATE_MAX_TOKENS, \"include_suites\": [\"atari\",\"craftaxfull\",\"nethack\"]}"
