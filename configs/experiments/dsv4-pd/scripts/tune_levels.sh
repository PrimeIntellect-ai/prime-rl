#!/bin/bash
# Run levels against already-running tuning jobs: for each level, start every arm's eval for the hold as a fresh burst
# and stop it; cancel the jobs at the end.
#
#   tune_levels.sh <level:minutes>[,...] <config-name>=<jobid> [...]
set -uo pipefail
STEPS=$1; shift
E=/home/phi/exp/dsv4-pd
log() { echo "$(date -u +%H:%M) $*"; }
declare -A JOB
for arm in "$@"; do JOB[${arm%=*}]=${arm#*=}; done
router() { grep -m1 '^HOSTNAMES=' $(ls -t $E/outputs/tune_$1/launcher/logs/job_*.log | head -1) | cut -d= -f2 | cut -d' ' -f1; }
for STEP in ${STEPS//,/ }; do
    LEVEL=${STEP%%:*}
    for cfg in "${!JOB[@]}"; do $E/scripts/tune_arm.sh tune_$cfg tune-$cfg-L$LEVEL "$STEP" 2>&1 | grep -v Warning; done
    for cfg in "${!JOB[@]}"; do
        until grep -q "=== schedule done" $E/logs/driver_tune-$cfg-L$LEVEL.log 2>/dev/null; do
            squeue -h -j ${JOB[$cfg]} 2>/dev/null | grep -q . || { log "$cfg: job ended mid-level"; break; }
            sleep 30
        done
        log "$cfg: level $LEVEL done"
    done
    for cfg in "${!JOB[@]}"; do
        SSH_AUTH_SOCK= timeout 120 ssh "$(router $cfg)" "tmux send-keys -t eval-tune-$cfg-L$LEVEL C-c; for i in \$(seq 1 40); do pgrep -f 'PRL::[E]val' >/dev/null || break; sleep 15; done; \
            pkill -9 -f 'PRL::[E]val'; tmux kill-session -t eval-tune-$cfg-L$LEVEL; tmux kill-session -t driver-tune-$cfg-L$LEVEL" </dev/null 2>&1 | grep -v Warning
        log "$cfg: level $LEVEL eval stopped"
    done
done
for cfg in "${!JOB[@]}"; do scancel ${JOB[$cfg]}; log "$cfg: job ${JOB[$cfg]} cancelled"; done
log "round done"
