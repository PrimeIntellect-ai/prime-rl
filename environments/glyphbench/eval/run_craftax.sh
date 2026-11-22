#!/usr/bin/env bash
# Craftax wrapper eval using the same verifiers/memory/Qwen3.5 defaults as
# eval/run_debug.sh. Override MAX_TURNS for short smoke tests; leave it unset
# for the env-native full horizon.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${GLYPHBENCH_CRAFTAX_VENV:-$HOME/.cache/glyphbench-craftax-eval}}"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] || [ "${GLYPHBENCH_CRAFTAX_SYNC:-0}" = "1" ]; then
    uv sync --frozen --extra craftax --extra eval
fi

# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

export TASK_IDS=${TASK_IDS:-'["glyphbench/craftaxfull-v0"]'}
export NUM_EPISODES=${NUM_EPISODES:-1}
export ROLLOUTS_PER_EXAMPLE=${ROLLOUTS_PER_EXAMPLE:-1}
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---gdn-prefill-backend triton}"

exec bash "$REPO_ROOT/eval/run_debug.sh"
