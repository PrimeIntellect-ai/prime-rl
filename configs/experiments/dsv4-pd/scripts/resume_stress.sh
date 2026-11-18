#!/bin/bash
# Start each arm's level loop as soon as its inference job is up (16/16 engines), without waiting for the others.
#
#   resume_stress.sh <level:minutes,...>
LEVELS=$1
E=/home/phi/exp/dsv4-pd
up() { local L=$E/outputs/stress_$1/logs/attempt_1/inference; [ "$(grep -l 'startup complete' $L/node_*.log 2>/dev/null | wc -l)" -ge 16 ]; }
pending="agg pd pdmc"
while [ -n "$pending" ]; do
    for a in $pending; do
        if grep -lq Traceback $E/outputs/stress_$a/logs/attempt_1/inference/*.log 2>/dev/null; then
            echo "$(date -u +%H:%M) $a: engine traceback, not starting it"; pending=${pending/$a/}
        elif up $a; then
            echo "$(date -u +%H:%M) $a up, starting its loop: $LEVELS"
            tmux new-session -d -s loop-$a "$E/scripts/level_loop.sh $a $LEVELS 2>&1 | tee -a $E/logs/loop_$a.log"
            pending=${pending/$a/}
        fi
    done
    pending=$(echo $pending)
    [ -n "$pending" ] && sleep 60
done
echo "$(date -u +%H:%M) all loops started"
