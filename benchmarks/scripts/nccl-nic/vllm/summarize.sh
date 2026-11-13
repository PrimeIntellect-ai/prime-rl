#!/usr/bin/env bash
# Usage: summarize.sh <cell out root>
# Prints one markdown row per layout and arm from measured.json and the driver and server logs.
set -uo pipefail

root=$1
echo "| layout | arm | commit | NCCL | out tok/s | req/s | TTFT p50/p99 ms | TPOT p50/p99 ms | E2E p50/p99 ms | shutdown |"
echo "|---|---|---|---|---|---|---|---|---|---|"
for dir in "$root"/*/*/; do
    dir=${dir%/}
    arm=$(basename "$dir")
    layout=$(basename "$(dirname "$dir")")
    driver=$(cat "$dir"/driver.log "$dir"/node*/driver.log 2>/dev/null)
    commit=$(grep -m1 -oE '\(([0-9a-f]{7,})\), layout' <<< "$driver" | grep -oE '[0-9a-f]{7,}')
    nccl=$(cat "$dir"/server.log "$dir"/node*/server.log 2>/dev/null | grep -oE 'NCCL version [0-9.]+' | sort -u | awk '{print $3}' | paste -sd/)
    shutdown=$(grep -oE 'RESULT [A-Z_]+: .*' <<< "$driver" | sed -E 's/RESULT ([A-Z_]+): inference exit code ([0-9]+), processes gone at ([^ ]+) s, total ([^ ]+) s/\1 exit \2, procs \3 s, GPU \4 s/' | paste -sd ';')
    if [[ -f $dir/measured.json ]]; then
        metrics=$(jq -r '[(.output_throughput|round), (.request_throughput*100|round/100), "\(.median_ttft_ms|round)/\(.p99_ttft_ms|round)", "\(.median_tpot_ms*10|round/10)/\(.p99_tpot_ms*10|round/10)", "\(.median_e2el_ms|round)/\(.p99_e2el_ms|round)"] | join(" | ")' "$dir/measured.json")
    else
        metrics="n/a | n/a | n/a | n/a | n/a"
    fi
    echo "| $layout | $arm | $commit | $nccl | $metrics | $shutdown |"
done
