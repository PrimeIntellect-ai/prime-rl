#!/usr/bin/env bash
# Run per-call-site selective AC arms of the DSv4 proxy back to back on one node, with balanced routing.
# Usage: sac_targets_ab.sh <jobid> <layers> <run name prefix> <arm,arm,...> [trainer CLI overrides...]
# Arms:
#   full: full AC
#   fit:  keep mHC pre, o_a_proj's bmm, and the FP8 GEMM outputs of o_b_proj, q_a_proj, kv_proj and the shared expert
#   qb:   fit with q_b_proj's FP8 GEMM output kept instead of q_a_proj's
set -euo pipefail

JOBID=$1
LAYERS=$2
PREFIX=$3
ARMS=$4
shift 4

HERE=$(cd "$(dirname "$0")" && pwd)
COMMON='"prime_rl::dsv4_mhc_pre", "aten::bmm@self_attn.o_a_proj", "prime_rl::fp8_blockwise_mm@self_attn.o_b_proj", "prime_rl::fp8_blockwise_mm@self_attn.kv_proj", "prime_rl::fp8_blockwise_mm@mlp.shared_expert"'
declare -A TARGETS=(
    [fit]="[$COMMON, \"prime_rl::fp8_blockwise_mm@self_attn.q_a_proj\"]"
    [qb]="[$COMMON, \"prime_rl::fp8_blockwise_mm@self_attn.q_b_proj\"]"
)

status=0
for arm in ${ARMS//,/ }; do
    args=(--model.debug.num-layers "$LAYERS" --model.debug.force-balanced-routing)
    if [[ $arm != full ]]; then
        args+=(--model.ac.mode selective --model.ac.targets "${TARGETS[$arm]}")
    fi
    "$HERE/run_arm.sh" "$JOBID" "$PREFIX-$arm" "${args[@]}" "$@" || { echo "arm $arm failed" >&2; status=1; }
done
exit $status
