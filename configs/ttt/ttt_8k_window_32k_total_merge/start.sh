#!/usr/bin/env bash
set -euo pipefail

PRIME_RL_ROOT="${PRIME_RL_ROOT:-/shared/prime-rl}"
DATASETS_CACHE="${HF_DATASETS_CACHE:-/shared/huggingface/datasets}"
DATASET_LOCK_PATTERN="*R2E*Gym*"
CONFIG_DIR="${PRIME_RL_ROOT}/configs/ttt/ttt_8k_window_32k_total_merge"

source "${PRIME_RL_ROOT}/.env"
cd "${PRIME_RL_ROOT}"

while IFS= read -r -d '' lockfile; do
    if ! fuser "$lockfile" >/dev/null 2>&1; then
        echo "Removing stale lock: $lockfile"
        rm -f "$lockfile"
    else
        echo "Lock held by active process, skipping: $lockfile"
    fi
done < <(find "$DATASETS_CACHE" -name "*.lock" -path "$DATASET_LOCK_PATTERN" -print0 2>/dev/null)

uv sync --all-extras

uv run prime sandbox delete --label ttt-prod-cluster --yes
uv run rl @ "${CONFIG_DIR}/prod.toml" \
  --orchestrator.heartbeat.url "${BETTER_STACK_URL_ORCH}" \
  --trainer.heartbeat.url "${BETTER_STACK_URL_TRAIN}"
