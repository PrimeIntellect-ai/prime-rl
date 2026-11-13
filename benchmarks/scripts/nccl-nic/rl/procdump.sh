#!/bin/bash
# Usage: procdump.sh <pid...>
# Prints status, per-thread comm, wchan and state, raw task stat lines and the kernel stack (if readable) of each pid.
for pid in "$@"; do
    [ -d "/proc/$pid" ] || continue
    echo "== pid $pid cmdline: $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-300)"
    grep -E "^(Name|State|PPid|Threads):" "/proc/$pid/status" 2>/dev/null
    for task in /proc/"$pid"/task/*; do
        echo "task $(basename "$task") comm=$(cat "$task/comm" 2>/dev/null) wchan=$(cat "$task/wchan" 2>/dev/null) state=$(sed 's/.*) //' "$task/stat" 2>/dev/null | cut -d' ' -f1)"
    done
    echo "-- raw task stat"
    cat /proc/"$pid"/task/*/stat 2>/dev/null
    echo "-- kernel stack"
    cat "/proc/$pid/stack" 2>&1 | head -40
done
