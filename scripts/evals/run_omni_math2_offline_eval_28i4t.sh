#!/usr/bin/env bash
set -euo pipefail

root="${PRIME_RL_ROOT:-/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl}"
run_root="${OFFLINE_EVAL_RUN_ROOT:?OFFLINE_EVAL_RUN_ROOT must point at the run_default directory}"
offline_output="${OFFLINE_EVAL_OUTPUT_DIR:-$(dirname "$run_root")/offline_eval_600x8_all_ckpts}"
arm="${OFFLINE_EVAL_ARM:?OFFLINE_EVAL_ARM must name the eval arm}"

cd "$root"

cmd=(
    uv run --no-sync python -m prime_rl.entrypoints.launch offline-eval
    --in-allocation
    --arm "$arm"
    --run-root "$run_root"
    --output-dir "$offline_output"
    --nodes "${OFFLINE_EVAL_NODE_COUNT:-${OFFLINE_EVAL_NODES:-8}}"
    --gpus-per-node "${OFFLINE_EVAL_GPUS_PER_NODE:-4}"
    --dp-per-node "${OFFLINE_EVAL_DP_PER_NODE:-4}"
    --tp "${OFFLINE_EVAL_TP:-1}"
    --api-server-count "${OFFLINE_EVAL_API_SERVER_COUNT:-4}"
    --dp-local "${OFFLINE_EVAL_DP_LOCAL:-4}"
    --port "${OFFLINE_EVAL_PORT:-9800}"
    --backend-port "${OFFLINE_EVAL_BACKEND_PORT:-9900}"
    --gpu-memory-utilization "${OFFLINE_EVAL_GPU_MEMORY_UTILIZATION:-0.95}"
    --max-model-len "${OFFLINE_EVAL_MAX_MODEL_LEN:-16384}"
    --max-num-seqs "${OFFLINE_EVAL_MAX_NUM_SEQS:-192}"
    --max-num-batched-tokens "${OFFLINE_EVAL_MAX_NUM_BATCHED_TOKENS:-65536}"
    --driver-node-count "${OFFLINE_EVAL_DRIVER_NODE_COUNT:-0}"
    --router-policy "${PRIME_RL_VLLM_ROUTER_POLICY:-round_robin}"
    --compare-output "${OFFLINE_EVAL_COMPARE_OUTPUT:-$root/outputs/omni_math2_rlvr_canary/offline_eval_comparison_20260512.md}"
    --root "$root"
    --patched-verifiers "${PATCHED_VERIFIERS_PATH:-/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs}"
    --omni-env-path "$root/environments/omni_math2_singleturn"
)

if [[ -n "${OFFLINE_EVAL_WEIGHTS_ROOT:-}" ]]; then
    cmd+=(--weights-root "$OFFLINE_EVAL_WEIGHTS_ROOT")
fi
if [[ -n "${OFFLINE_EVAL_ROUTER_PORT:-}" ]]; then
    cmd+=(--router-port "$OFFLINE_EVAL_ROUTER_PORT")
fi
if [[ -n "${OFFLINE_EVAL_BASE_MODEL:-}" ]]; then
    cmd+=(--base-model "$OFFLINE_EVAL_BASE_MODEL")
fi
if [[ -n "${OFFLINE_EVAL_STEPS:-}" ]]; then
    cmd+=(--steps "$OFFLINE_EVAL_STEPS")
else
    cmd+=(
        --step-interval "${OFFLINE_EVAL_STEP_INTERVAL:-25}"
        --min-step "${OFFLINE_EVAL_MIN_STEP:-25}"
        --max-step "${OFFLINE_EVAL_MAX_STEP:-100}"
    )
fi
cmd+=(--wait-step "${OFFLINE_EVAL_WAIT_STEP:-${OFFLINE_EVAL_MAX_STEP:-100}}")
if [[ -n "${OFFLINE_EVAL_NUM_EXAMPLES:-}" ]]; then
    cmd+=(--num-examples "$OFFLINE_EVAL_NUM_EXAMPLES")
fi
if [[ -n "${OFFLINE_EVAL_ROLLOUTS_PER_EXAMPLE:-}" ]]; then
    cmd+=(--rollouts-per-example "$OFFLINE_EVAL_ROLLOUTS_PER_EXAMPLE")
fi
if [[ -n "${OFFLINE_EVAL_MAX_CONCURRENCY:-}" ]]; then
    cmd+=(--max-concurrency "$OFFLINE_EVAL_MAX_CONCURRENCY")
fi
if [[ -n "${OFFLINE_EVAL_SCORE_MAX_CONCURRENCY:-}" ]]; then
    cmd+=(--score-max-concurrency "$OFFLINE_EVAL_SCORE_MAX_CONCURRENCY")
fi
if [[ -n "${OFFLINE_EVAL_MAX_RETRIES:-}" ]]; then
    cmd+=(--max-retries "$OFFLINE_EVAL_MAX_RETRIES")
fi
if [[ -n "${OFFLINE_EVAL_KS:-}" ]]; then
    cmd+=(--ks "$OFFLINE_EVAL_KS")
fi
if [[ "${PRIME_RL_DISABLE_VLLM_ROUTER:-0}" == "1" ]]; then
    cmd+=(--disable-router)
fi

exec "${cmd[@]}"
