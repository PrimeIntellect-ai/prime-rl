#!/usr/bin/env bash
# Full eval over the active subset of GlyphBench.
#
# By default this excludes the archival ``atari`` (long-horizon originals),
# ``craftaxfull`` (open-ended Crafter), ``nethack``, and ``agentick`` suites.
# To include them, run ``eval/run_archival.sh`` or pass ``EXCLUDE_SUITES='[]'``.
#
# Prereq: a vLLM server at $VLLM_BASE_URL serving the target model.
# See eval/README.md for the serving command and pinned-runtime settings.
#
# Budget arithmetic for tasks with memory enabled:
#   action call:  prompt (sys+obs) + 4096 action_output_tokens   <= max-model-len
#   memory call:  prompt (sys+obs+action_tag+lean_user) + 4096   <= max-model-len
#                 where lean_user re-injects the action reasoning (<=4096 tok)
#                 plus the next-obs grid (~1500 tok) and the [Memory Update]
#                 instruction.
# 65536 covers the worst case: ~8K prompt + 4K reasoning re-injection +
# 1.5K next-obs + 0.5K wrappers + 4K memory output budget = ~18K, with
# margin for craftax-sized grids and longer system prompts.
#
# After it finishes, view results with: prime eval tui

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
EPISODES=${EPISODES:-$GLYPHBENCH_EVAL_EPISODES}
ROLLOUTS_PER_EXAMPLE=${ROLLOUTS_PER_EXAMPLE:-$GLYPHBENCH_EVAL_ROLLOUTS_PER_EXAMPLE}
SEED=${SEED:-$GLYPHBENCH_EVAL_SEED}
N_FRAMES=${N_FRAMES:-$GLYPHBENCH_EVAL_N_FRAMES}
MAX_TOKENS=${MAX_TOKENS:-$GLYPHBENCH_EVAL_MAX_TOKENS}
MEMORY_UPDATE_MAX_TOKENS=${MEMORY_UPDATE_MAX_TOKENS:-$GLYPHBENCH_EVAL_MEMORY_UPDATE_MAX_TOKENS}
EXCLUDE_SUITES=${EXCLUDE_SUITES:-$GLYPHBENCH_EVAL_EXCLUDE_SUITES}
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
  -a "{\"num_episodes\": $EPISODES, \"n_frames\": $N_FRAMES, \"seed\": $SEED, \"max_output_tokens\": $MAX_TOKENS, \"memory_update_max_tokens\": $MEMORY_UPDATE_MAX_TOKENS, \"exclude_suites\": $EXCLUDE_SUITES}"
