#!/bin/bash
# Usage: warnings.sh <run dir>
# Prints the distinct warning, error and NCCL lines of a run's component logs with timestamps, pids and numbers normalized.
cd "$1/bench" || exit 1
for f in ranks/rank_0.log orchestrator.log vllm_0.log router.log env_pool.log torchrun.log; do
    grep -hiE "warn|error|traceback|exception|nccl|mismatch|fail" "$f" 2>/dev/null \
        | sed -E -e 's/^[0-9.]+ //' -e 's/\x1b\[[0-9;]*m//g' -e 's/[0-9]+/N/g' \
        | cut -c1-220 | sort -u | sed "s|^|$f: |"
done
