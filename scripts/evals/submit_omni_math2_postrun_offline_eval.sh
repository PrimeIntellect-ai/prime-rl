#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "usage: $0 AFTER_JOB_ID ARM RUN_ROOT OFFLINE_OUTPUT_DIR" >&2
    exit 2
fi

after_job_id="$1"
arm="$2"
run_root="$3"
offline_output="$4"
root="${PRIME_RL_ROOT:-/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl}"

cd "$root"

cmd=(
    uv run --no-sync python -m prime_rl.entrypoints.launch offline-eval
    --after-job-id "$after_job_id"
    --arm "$arm"
    --run-root "$run_root"
    --output-dir "$offline_output"
    --nodes "${OFFLINE_EVAL_NODE_COUNT:-${OFFLINE_EVAL_NODES:-8}}"
    --sbatch-nodes "${OFFLINE_EVAL_SBATCH_NODES:-8}"
    --gpus-per-node "${OFFLINE_EVAL_GPUS_PER_NODE:-4}"
    --driver-node-count "${OFFLINE_EVAL_DRIVER_NODE_COUNT:-0}"
    --router-policy "${PRIME_RL_VLLM_ROUTER_POLICY:-round_robin}"
    --partition "${OFFLINE_EVAL_PARTITION:-workq}"
    --account "${OFFLINE_EVAL_ACCOUNT:-brics.a6r}"
    --time-limit "${OFFLINE_EVAL_TIME_LIMIT:-06:00:00}"
    --dependency-type "${OFFLINE_EVAL_DEPENDENCY_TYPE:-afterany}"
    --postrun-settle-seconds "${OFFLINE_EVAL_POSTRUN_SETTLE_SECONDS:-120}"
    --compare-output "${OFFLINE_EVAL_COMPARE_OUTPUT:-$root/outputs/omni_math2_rlvr_canary/offline_eval_comparison_20260512.md}"
    --root "$root"
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
