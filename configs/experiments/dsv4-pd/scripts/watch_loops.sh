#!/bin/bash
# Return when every arm in $ARMS (default: agg pd) finished level $1, or on an anomaly. Only reads loop-log lines
# written after this watcher starts.
LEVEL=$1
ARMS=${ARMS:-agg pd}
E=/home/phi/exp/dsv4-pd
declare -A base
for a in $ARMS; do base[$a]=$(wc -l < $E/logs/loop_$a.log); done
new_lines() { tail -n +$(( ${base[$1]} + 1 )) $E/logs/loop_$1.log; }
while true; do
    done_n=0
    for a in $ARMS; do
        tb=$(grep -l Traceback $E/outputs/stress_$a/logs/attempt_1/inference/*.log 2>/dev/null | head -1)
        [ -n "$tb" ] && { echo "ANOMALY $a: engine traceback in $tb"; grep -h -A3 Traceback $tb | tail -5 | cut -c1-250; exit 1; }
        n=$(grep -c ERROR $E/outputs/stress-$a-L$LEVEL/logs/latest/eval.log 2>/dev/null)
        [ "${n:-0}" -gt 20 ] && { echo "ANOMALY $a: $n eval ERROR lines"; exit 1; }
        new_lines $a | grep -qE "driver session gone|job .* ended" && { echo "ANOMALY $a:"; new_lines $a | tail -3; exit 1; }
        new_lines $a | grep -q "level $LEVEL: eval stopped" && done_n=$((done_n + 1))
    done
    [ $done_n -ge $(echo $ARMS | wc -w) ] && { date; echo "level $LEVEL done in: $ARMS"; exit 0; }
    sleep 30
done
