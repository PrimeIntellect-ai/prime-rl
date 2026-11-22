#!/usr/bin/env bash
# NetHack full eval using the GlyphBench wrapper in the local NLE fork.
# Uses an isolated uv environment to keep the NLE/eval dependencies stable.
# Leave MAX_TURNS unset for the native million-step NetHack horizon. Set it
# only for deliberate smoke tests.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${GLYPHBENCH_NETHACK_VENV:-$HOME/.cache/glyphbench-nethack-eval}}"

if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] || [ "${GLYPHBENCH_NETHACK_SYNC:-0}" = "1" ]; then
    uv sync --frozen --extra nethack --extra eval
fi

# Activate directly after syncing. Some GPU workers do not have the system
# parser generators needed to rebuild NLE, so avoid an implicit `uv run` sync
# in the actual eval process.
# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
eval "$(python "$REPO_ROOT/src/glyphbench/eval_defaults.py" shell)"

export TASK_IDS=${TASK_IDS:-'["glyphbench/nethack-full-v0"]'}
export NUM_EPISODES=${NUM_EPISODES:-1}
export ROLLOUTS_PER_EXAMPLE=${ROLLOUTS_PER_EXAMPLE:-$GLYPHBENCH_EVAL_ROLLOUTS_PER_EXAMPLE}

MODEL=${MODEL:-$GLYPHBENCH_EVAL_MODEL}
BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}
AUTO_START_VLLM=${AUTO_START_VLLM:-0}
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---gdn-prefill-backend triton}"

is_vllm_up() {
    curl -fs --max-time 3 "${BASE_URL%/v1}/v1/models" > /dev/null 2>&1
}

if ! is_vllm_up && [ "$AUTO_START_VLLM" != "1" ]; then
    cat >&2 <<EOF
!!! No vLLM server reachable at $BASE_URL.

Start one with the isolated NetHack eval environment:
    cd $REPO_ROOT
    export UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT"
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
    set -a; [ -f .env ] && source .env; set +a
    vllm serve $MODEL --port 8000 --max-model-len $GLYPHBENCH_EVAL_MAX_MODEL_LEN --enforce-eager \\
        --max-num-batched-tokens 8192 --enable-chunked-prefill \\
        --compilation-config '{"custom_ops":["-silu_and_mul"]}'

Then run:
    bash eval/run_nethack.sh

Or let the script start vLLM:
    AUTO_START_VLLM=1 bash eval/run_nethack.sh
EOF
    exit 2
fi

exec bash "$REPO_ROOT/eval/run_debug.sh"
