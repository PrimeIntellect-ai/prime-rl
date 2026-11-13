#!/usr/bin/env bash
# Usage: MAIN_CHECKOUT=<dir> FIX_CHECKOUT=<dir> run_cell.sh <out root> <deadline, any `date -d` string> <min seconds per launch> <layout toml> [layout toml ...]
# Runs run_one.sh for each layout with the arms in $ARMS (default: main then fix), skipping any launch that cannot finish before the deadline.
# MAIN_CHECKOUT and FIX_CHECKOUT are prime-rl checkouts with synced .venvs; the main checkout's vllm also drives the benchmark client.
set -uo pipefail

here=$(dirname "$(readlink -f "$0")")
root=$1
deadline=$(date -d "$2" +%s)
min_launch_s=$3
shift 3
read -ra arms <<< "${ARMS:-main fix}"
declare -A checkouts=(
    [main]=${MAIN_CHECKOUT:?set MAIN_CHECKOUT to the main arm checkout}
    [fix]=${FIX_CHECKOUT:?set FIX_CHECKOUT to the fix arm checkout}
)

for layout in "$@"; do
    name=$(basename "$layout" .toml)
    for arm in "${arms[@]}"; do
        if (( $(date +%s) + min_launch_s > deadline )); then
            echo "$(date -u +%FT%T) SKIP $name $arm: fewer than $min_launch_s s before the deadline"
            continue
        fi
        echo "$(date -u +%FT%T) START $name $arm"
        BENCH_CHECKOUT=${checkouts[main]} "$here/run_one.sh" "${checkouts[$arm]}" "$layout" "$root/$name/$arm"
    done
done
