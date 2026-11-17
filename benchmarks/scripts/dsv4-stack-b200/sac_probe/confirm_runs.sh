#!/usr/bin/env bash
set -uo pipefail
source $INVESTIGATION/env.sh
unset PYTHONPATH SAC_PROBE_LOG
cd $WORKTREES/prime-rl-arm-4358
COMMON=(--max-steps 3 --model.debug.force-balanced-routing --model.debug.num-layers 6 --no-monitors.prime --model.name "$DSV4_CKPT" --model.debug.random-init --model.dsv4-cp-kv-exchange halo)
CANDIDATE=$(cat $INVESTIGATION/inputs/sac_targets.json.candidate)
benchmarks/scripts/dsv4-perf-proxy/run_arm.sh $JOB sac-confirm-1 "${COMMON[@]}" --model.ac.mode selective --model.ac.targets "$CANDIDATE"
echo "sac-confirm-1 exit=$?"
benchmarks/scripts/dsv4-perf-proxy/run_arm.sh $JOB full-confirm-1 "${COMMON[@]}" --model.ac.mode full
echo "full-confirm-1 exit=$?"
benchmarks/scripts/dsv4-perf-proxy/run_arm.sh $JOB sac-nomm-1 "${COMMON[@]}" --model.ac.mode selective --model.ac.targets '["prime_rl::dsv4_mhc_pre", "aten::bmm@self_attn.o_a_proj"]'
echo "sac-nomm-1 exit=$?"
