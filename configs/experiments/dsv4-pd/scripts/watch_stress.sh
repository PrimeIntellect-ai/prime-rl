#!/bin/bash
# Return when every arm's driver has started level $1, or on an anomaly (eval errors, stale driver log, engine traceback).
LEVEL=$1
E=/home/phi/exp/dsv4-pd
while true; do
    for a in agg pd pdmc; do
        d=$E/logs/driver_stress-$a${SUFFIX:-}.log
        n=$(grep -c "ERROR" $E/outputs/stress-$a/logs/latest/eval.log 2>/dev/null)
        [ "${n:-0}" -gt 20 ] && { echo "ANOMALY $a: $n eval ERROR lines"; grep ERROR $E/outputs/stress-$a/logs/latest/eval.log | tail -3 | cut -c1-250; exit 1; }
        age=$(( $(date +%s) - $(stat -c %Y $d) ))
        [ $age -gt 300 ] && { echo "ANOMALY $a: driver log stale ${age}s"; tail -3 $d; exit 1; }
        tb=$(grep -l Traceback $E/outputs/stress_$a/logs/attempt_1/inference/*.log 2>/dev/null | head -1)
        [ -n "$tb" ] && { echo "ANOMALY $a: engine traceback in $tb"; grep -h -A3 Traceback $tb | tail -5 | cut -c1-250; exit 1; }
    done
    [ "$(grep -l "=== level $LEVEL " $E/logs/driver_stress-{agg,pd,pdmc}${SUFFIX:-}.log 2>/dev/null | wc -l)" -ge 3 ] && { date; echo "all arms at level $LEVEL"; exit 0; }
    sleep 30
done
