#!/bin/bash
# One tuning round: submit each arm's inference job (pinned to a nodelist when given), wait until all are up, then for
# each level run every arm's eval for the hold as a fresh burst and stop it; cancel the jobs at the end. Arms run in
# parallel within the round.
#
#   bidir_round.sh <level:minutes>[,<level:minutes>...] <config-name>[@<nodelist>] [...]
set -uo pipefail
STEPS=$1; shift
E=/home/phi/exp/dsv4-pd
WT=/home/phi/lib/prime-rl-bidir
log() { echo "$(date -u +%H:%M) $*"; }
# Jobs inherit the submitting environment: carry the credentials (HF_TOKEN, PRIME_*, WANDB) along.
source ~/.env
declare -A JOB HOST
for arm in "$@"; do
    cfg=${arm%@*}; nodes=""; [ "$arm" != "$cfg" ] && nodes=${arm#*@}
    rm -rf $E/outputs/tune_$cfg
    JOB[$cfg]=$(cd $WT && uv run --no-sync inference @ $E/configs/bidir/$cfg.toml ${nodes:+--slurm.nodelist "$nodes"} 2>&1 | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+$")
    log "$cfg: job ${JOB[$cfg]} on ${nodes:-any nodes}"
    [ -n "${JOB[$cfg]}" ] || { log "$cfg: submit failed"; exit 1; }
done
nodes_up() { local L=$E/outputs/tune_$1/logs/attempt_1/inference; local n; n=$(ls $L/node_*.log 2>/dev/null | grep -vc rank); \
    [ "$n" -gt 0 ] && [ "$(grep -l 'startup complete' $L/node_*.log 2>/dev/null | wc -l)" -ge $((n * 8)) ]; }
for cfg in "${!JOB[@]}"; do
    until nodes_up $cfg; do
        grep -lq Traceback $E/outputs/tune_$cfg/logs/attempt_1/inference/*.log 2>/dev/null && { log "$cfg: engine traceback"; exit 1; }
        squeue -h -j ${JOB[$cfg]} | grep -q . || { log "$cfg: job ended before startup"; exit 1; }
        sleep 30
    done
    log "$cfg: up"
done
router() { grep -m1 '^HOSTNAMES=' $(ls -t $E/outputs/tune_$1/launcher/logs/job_*.log | head -1) | cut -d= -f2 | cut -d' ' -f1; }
for STEP in ${STEPS//,/ }; do
    LEVEL=${STEP%%:*}
    for cfg in "${!JOB[@]}"; do $E/scripts/bidir_arm.sh tune_$cfg tune-$cfg-L$LEVEL "$STEP" 2>&1 | grep -v Warning; done
    for cfg in "${!JOB[@]}"; do
        until grep -q "=== schedule done" $E/logs/driver_tune-$cfg-L$LEVEL.log 2>/dev/null; do
            squeue -h -j ${JOB[$cfg]} | grep -q . || { log "$cfg: job ended mid-level"; break; }
            sleep 30
        done
        log "$cfg: level $LEVEL done"
    done
    for cfg in "${!JOB[@]}"; do
        SSH_AUTH_SOCK= timeout 120 ssh "$(router $cfg)" "tmux send-keys -t eval-tune-$cfg-L$LEVEL C-c; for i in \$(seq 1 40); do pgrep -f 'PRL::[E]val' >/dev/null || break; sleep 15; done; \
            pkill -9 -f 'PRL::[E]val'; tmux kill-session -t eval-tune-$cfg-L$LEVEL; tmux kill-session -t driver-tune-$cfg-L$LEVEL; tmux kill-session -t routerm-tune-$cfg-L$LEVEL" 2>&1 | grep -v Warning
        log "$cfg: level $LEVEL eval stopped"
    done
done
for cfg in "${!JOB[@]}"; do scancel ${JOB[$cfg]}; log "$cfg: job ${JOB[$cfg]} cancelled"; done
log "round done"
