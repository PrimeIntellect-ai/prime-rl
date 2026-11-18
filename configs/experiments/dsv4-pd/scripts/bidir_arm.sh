#!/bin/bash
# Token-in variant of tune_arm.sh for the bidirectional-KV branch (PR #4575): the eval rolls out through
# /inference/v1/generate with session ids (PRL_EVAL_TOKEN_IN=1), and the router's bidirectional counters are sampled.
# Start an eval + staircase driver against one tuning arm, on the arm's router node. The layout (prefill/decode node
# count, host order) comes from the arm's generated sbatch script and job log, so any P/D or multi-node shape works.
#
#   bidir_arm.sh <outputs-subdir, e.g. tune_pd_mbt32k> <run-name> <level:minutes,...>
set -euo pipefail
OUTNAME=$1; RUN=$2; SCHEDULE=$3
EXP=/home/phi/exp/dsv4-pd
WT=/home/phi/lib/prime-rl-bidir
OUT=$EXP/outputs/$OUTNAME
JOB_LOG=$(ls -t $OUT/launcher/logs/job_*.log | head -1)
read -ra HOSTS <<< "$(grep -m1 '^HOSTNAMES=' "$JOB_LOG" | cut -d= -f2)"
NP=$(grep -m1 '^export NUM_PREFILL_NODES=' $OUT/launcher/inference.sbatch | cut -d= -f2 || true)
ADMIN=(); ROLES=(); ENGINES=()
for i in "${!HOSTS[@]}"; do
    h=${HOSTS[$i]}
    if [ -z "$NP" ]; then role=agg; port=8100
    elif [ "$i" -lt "$NP" ]; then role=prefill; port=8100
    else role=decode; port=8200; fi
    for k in $(seq 0 7); do
        ADMIN+=("http://$h:$((port + k))/v1"); ENGINES+=("--engine" "http://$h:$((port + k))=$role")
        [ "$role" != agg ] && ROLES+=("$role")
    done
done
EVAL_CMD="uv run --no-sync eval @ $EXP/configs/stress/stress_eval.toml --run.name $RUN --client.base-url http://localhost:8000/v1 --client.admin-base-url ${ADMIN[*]}"
[ ${#ROLES[@]} -gt 0 ] && EVAL_CMD="$EVAL_CMD --inference-metrics-roles ${ROLES[*]}"
DRIVER_CMD="uv run --no-sync python $EXP/scripts/stress_driver.py --control http://localhost:8045 --schedule $SCHEDULE --out $EXP/stress/$RUN.jsonl ${ENGINES[*]}"
echo "$OUTNAME: ${#HOSTS[@]} nodes (prefill nodes: ${NP:-agg}), router ${HOSTS[0]}: $RUN, schedule $SCHEDULE"
SSH_AUTH_SOCK= ssh -o StrictHostKeyChecking=accept-new "${HOSTS[0]}" "tmux new-session -d -s eval-$RUN \"bash -lc 'source ~/.env; export PRL_EVAL_TOKEN_IN=1; cd $WT && $EVAL_CMD 2>&1 | tee $EXP/logs/eval_$RUN.log; exec bash'\" && \
    tmux new-session -d -s driver-$RUN \"bash -lc 'cd $WT && $DRIVER_CMD 2>&1 | tee $EXP/logs/driver_$RUN.log; exec bash'\""
# Router-side bidirectional counters (handles stored / used / dropped), once a minute.
SSH_AUTH_SOCK= ssh "${HOSTS[0]}" "tmux new-session -d -s routerm-$RUN '$EXP/scripts/router_metrics.sh $EXP/stress/router_$RUN.prom'"
