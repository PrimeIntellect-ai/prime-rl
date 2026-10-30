#!/bin/bash
# The 8-node A/B arms of the 2026-10-30 trainer benchmark, each run inside its own 8-node salloc
# allocation on the batches of resplit_replay.py. Arm E needs prime-mega-moe's deep_gemm, which its
# pre_run_command installs after the job's uv sync (DEEP_GEMM_WHEEL).
# Usage: MODEL=<hf id> REPLAY=<dir of replay sets> [DEEP_GEMM_WHEEL=<wheel>] ab_arms.sh <arm> <jobid>
set -euo pipefail
: "${REPLAY:?set REPLAY to the directory holding the dp8 and dp16 replay sets}"
ARM=$1
JOBID=$2
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
SHA=$(git -C "$REPO" rev-parse --short=7 HEAD)

case $ARM in
    A) args=(--trainer.data.replay "$REPLAY/b1024-mb10-dp8") ;;
    B) args=(--trainer.model.cp 4 --trainer.model.compile.fullgraph true --trainer.data.replay "$REPLAY/b1024-mb10-dp16") ;;
    D) args=(--trainer.model.compile.fullgraph true --trainer.data.replay "$REPLAY/b1024-mb10-dp8") ;;
    E)
        : "${DEEP_GEMM_WHEEL:?set DEEP_GEMM_WHEEL to prime-mega-moe's deep_gemm wheel}"
        args=(
            --trainer.model.compile.fullgraph true --trainer.data.replay "$REPLAY/b1024-mb10-dp8"
            --trainer.model.moe.dispatch.type mega_moe --trainer.model.moe.dispatch.precision mxfp8
            --trainer.model.ac.mode selective --trainer.model.ac.targets '["prime_rl::mega_moe_forward"]'
            --env-vars.UV_NO_SYNC 1
            --slurm.pre-run-command "uv pip install --no-deps --reinstall-package deep_gemm $DEEP_GEMM_WHEEL && uv run --no-sync python -c 'import deep_gemm; assert deep_gemm.mxfp8_mega_moe'"
        )
        ;;
    *) echo "unknown arm $ARM" >&2; exit 1 ;;
esac
"$HERE/launch_in_alloc.sh" "$JOBID" "$REPO" "$ARM-$SHA" "${args[@]}"
