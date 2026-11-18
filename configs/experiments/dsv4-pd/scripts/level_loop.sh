#!/bin/bash
# Run an arm's levels high to low, each as a fresh eval burst: start eval + driver pinned at the level, hold it,
# then stop the eval (its in-flight episodes are cancelled with it) before the next level.
#
#   [STOP_FIRST=<run>] level_loop.sh <arm> <level:minutes,...>
set -uo pipefail
ARM=$1; LEVELS=$2
EXP=/home/phi/exp/dsv4-pd
JOB_LOG=$(ls -t $EXP/outputs/stress_$ARM/launcher/logs/job_*.log | head -1)
H0=$(grep -m1 '^HOSTNAMES=' "$JOB_LOG" | cut -d= -f2 | cut -d' ' -f1)
JOB=$(basename "$JOB_LOG" .log | sed 's/^job_//')
# Stop an eval run and its driver. The eval renames itself PRL::Eval (one per router node); the bracket keeps
# pgrep from matching this ssh command itself.
stop_run() {
    local run=$1
    node "tmux send-keys -t eval-$run C-c"
    for _ in $(seq 1 40); do node "pgrep -f 'PRL::[E]val' >/dev/null && echo up" | grep -q up || break; sleep 15; done
    node "pkill -9 -f 'PRL::[E]val'; tmux kill-session -t eval-$run; tmux kill-session -t driver-$run" >/dev/null
}
node() { SSH_AUTH_SOCK= timeout 120 ssh -o StrictHostKeyChecking=accept-new "$H0" "$@" 2>&1 | grep -v Warning; }
[ -n "${STOP_FIRST:-}" ] && { echo "$(date -u +%H:%M) $ARM: stopping $STOP_FIRST"; stop_run "$STOP_FIRST"; }
for step in ${LEVELS//,/ }; do
    level=${step%%:*}; RUN=stress-$ARM-L$level
    echo "$(date -u +%H:%M) $ARM level $level: start $RUN"
    $EXP/scripts/run_stress_arm.sh $ARM $RUN "$step" | grep -v Warning | head -1
    until grep -q "=== schedule done" $EXP/logs/driver_$RUN.log 2>/dev/null; do
        sleep 30
        # "alive"/"gone" from the node; anything else (ssh failed, node lost) is retried, never read as gone.
        state=$(node "tmux has-session -t driver-$RUN 2>/dev/null && echo alive || echo gone" | grep -oE "alive|gone" | head -1)
        [ "$state" = gone ] && { echo "driver session gone"; break; }
        squeue -h -j "$JOB" 2>/dev/null | grep -q . || { echo "$(date -u +%H:%M) $ARM: inference job $JOB ended, stopping"; exit 1; }
    done
    echo "$(date -u +%H:%M) $ARM level $level: hold done, stopping eval"
    stop_run $RUN
    echo "$(date -u +%H:%M) $ARM level $level: eval stopped"
done
echo "$(date -u +%H:%M) $ARM: all levels done"
