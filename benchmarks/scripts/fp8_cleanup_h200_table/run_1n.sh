#!/bin/bash
# One single-node run inside its own allocation. Usage: run_1n.sh CONFIG ARM KIND REP (KIND: time or trace)
set -u
cfg=$1 arm=$2 kind=$3 rep=$4
name=fp8table-${cfg#sft_}-$arm-$kind-r$rep
source /home/garrett/github/PrimeIntellect-ai/prime-rl-feat-fp8-cleanup/benchmarks/scripts/fp8_cleanup_h200_table/env.sh $arm $name
extra=(); [ "$kind" = trace ] && extra=(--max-steps 5 --trace-path $HOME/tmp/profiling/fp8-cleanup-h200-table/traces/$name)
rm -rf /tmp/$USER/$name; mkdir -p /tmp/$USER/$name
cd $WT
echo "=== $name $(hostname) start $(date -u +%T)"
uv run --no-sync sft @ $C/$cfg.toml "${OVERLAY[@]}" --run.name $name --clean "${extra[@]}" > $C/results/$name.log 2>&1
echo "=== $name exit $? $(date -u +%T)"
