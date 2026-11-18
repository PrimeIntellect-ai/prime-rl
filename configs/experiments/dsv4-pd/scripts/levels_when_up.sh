#!/bin/bash
# Wait until every given tuning job's engines are up, then run tune_levels.sh against them.
#
#   levels_when_up.sh <level:minutes>[,...] <config-name>=<jobid> [...]
E=/home/phi/exp/dsv4-pd
STEPS=$1; shift
for arm in "$@"; do
    cfg=${arm%=*}; job=${arm#*=}; L=$E/outputs/tune_$cfg/logs/attempt_1/inference
    until n=$(ls $L/node_*.log 2>/dev/null | grep -vc rank) && [ "$n" -gt 0 ] && [ "$(grep -l 'startup complete' $L/node_*.log 2>/dev/null | wc -l)" -ge $((n * 8)) ]; do
        squeue -h -j $job 2>/dev/null | grep -q . || { echo "$(date -u +%H:%M) $cfg: job $job ended before startup"; exit 1; }
        sleep 30
    done
    echo "$(date -u +%H:%M) $cfg: up"
done
exec $E/scripts/tune_levels.sh "$STEPS" "$@"
