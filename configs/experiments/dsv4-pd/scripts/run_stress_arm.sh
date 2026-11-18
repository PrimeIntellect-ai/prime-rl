#!/bin/bash
# Start one stress arm's eval + staircase driver in tmux on the arm's router node (its first SLURM host).
#
#   run_stress_arm.sh <arm: agg|pd|pdmc> <run-name> <schedule level:minutes,...> [driver-tag]
# With a driver tag, only (re)starts the driver against the running eval <run-name>, writing stress/<driver-tag>.jsonl.
set -euo pipefail
ARM=$1; RUN=$2; SCHEDULE=$3; TAG=${4:-}
EXP=/home/phi/exp/dsv4-pd
WT=/home/phi/lib/prime-rl-pd
OUT=$EXP/outputs/stress_$ARM
JOB_LOG=$(ls -t $OUT/launcher/logs/job_*.log | head -1)
read -ra HOSTS <<< "$(grep -m1 '^HOSTNAMES=' "$JOB_LOG" | cut -d= -f2)"
H0=${HOSTS[0]}; H1=${HOSTS[1]}

ADMIN=(); ROLES=(); ENGINES=()
if [ "$ARM" = agg ]; then
    for h in "$H0" "$H1"; do for k in $(seq 0 7); do
        ADMIN+=("http://$h:$((8100 + k))/v1"); ENGINES+=("--engine" "http://$h:$((8100 + k))=agg")
    done; done
else
    for k in $(seq 0 7); do
        ADMIN+=("http://$H0:$((8100 + k))/v1"); ROLES+=(prefill); ENGINES+=("--engine" "http://$H0:$((8100 + k))=prefill")
    done
    for k in $(seq 0 7); do
        ADMIN+=("http://$H1:$((8200 + k))/v1"); ROLES+=(decode); ENGINES+=("--engine" "http://$H1:$((8200 + k))=decode")
    done
fi

EVAL_CMD="uv run --no-sync eval @ $EXP/configs/stress/stress_eval.toml --run.name $RUN --client.base-url http://localhost:8000/v1 --client.admin-base-url ${ADMIN[*]}"
[ ${#ROLES[@]} -gt 0 ] && EVAL_CMD="$EVAL_CMD --inference-metrics-roles ${ROLES[*]}"
DRIVER_CMD="uv run --no-sync python $EXP/scripts/stress_driver.py --control http://localhost:8045 --schedule $SCHEDULE --out $EXP/stress/${TAG:-$RUN}.jsonl ${ENGINES[*]}"

mkdir -p $EXP/stress $EXP/logs
echo "$ARM on $H0 (router) / $H1: $RUN, schedule $SCHEDULE${TAG:+, driver only -> $TAG}"
if [ -n "$TAG" ]; then
    SSH_AUTH_SOCK= ssh -o StrictHostKeyChecking=accept-new "$H0" "tmux kill-session -t driver-$RUN 2>/dev/null; \
        tmux new-session -d -s driver-$RUN \"bash -lc 'cd $WT && $DRIVER_CMD 2>&1 | tee $EXP/logs/driver_$TAG.log; exec bash'\" && tmux ls"
    exit 0
fi
SSH_AUTH_SOCK= ssh -o StrictHostKeyChecking=accept-new "$H0" "tmux new-session -d -s eval-$RUN \"bash -lc 'source ~/.env; cd $WT && $EVAL_CMD 2>&1 | tee $EXP/logs/eval_$RUN.log; exec bash'\" && \
    tmux new-session -d -s driver-$RUN \"bash -lc 'cd $WT && $DRIVER_CMD 2>&1 | tee $EXP/logs/driver_$RUN.log; exec bash'\" && tmux ls"
