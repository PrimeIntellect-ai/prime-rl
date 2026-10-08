#!/usr/bin/env bash
# Every launch of the Dion Muon EP local-expert A/B, in order. Arm A pins Dion 6f9242d, arm B pins
# the fix. Each arm runs from its own worktree detached at the arm's commit, with its own venv.
set -euo pipefail

H=benchmarks/scripts/dion-muon-ep
A=$HOME/github/PrimeIntellect-ai/prime-rl-dion-ab-a
B=$HOME/github/PrimeIntellect-ai/prime-rl-dion-ab-b
P1_HOST=prime-nebius-puku-h200-gpu-062
P2_HOSTS=prime-nebius-puku-h200-gpu-064,prime-nebius-puku-h200-gpu-043
F8_HOSTS=prime-nebius-puku-h200-gpu-022,prime-nebius-puku-h200-gpu-024,prime-nebius-puku-h200-gpu-025,prime-nebius-puku-h200-gpu-027,prime-nebius-puku-h200-gpu-043,prime-nebius-puku-h200-gpu-048,prime-nebius-puku-h200-gpu-062,prime-nebius-puku-h200-gpu-064
TRACES=$HOME/tmp/profiling/dion-muon-ep/traces
MEMORY=$HOME/tmp/profiling/dion-muon-ep/memory

p1() {
    for arm in a b; do
        dir=$A; [ $arm = b ] && dir=$B
        for rep in 1 2; do $dir/$H/run_cell.sh $dir p1 dion-ab-p1-$arm-$rep $P1_HOST; done
        $dir/$H/run_cell.sh $dir p1 dion-ab-p1-$arm-mem $P1_HOST --max-steps 5 \
            --memory-profiler-path $MEMORY/p1-$arm
    done
}

p2() {
    for arm in a b; do
        dir=$A; [ $arm = b ] && dir=$B
        for rep in 1 2; do $dir/$H/run_cell.sh $dir p2 dion-ab-p2-$arm-$rep $P2_HOSTS; done
        $dir/$H/run_cell.sh $dir p2 dion-ab-p2-$arm-trace $P2_HOSTS --max-steps 5 \
            --trace-path $TRACES/p2-$arm
        $dir/$H/run_cell.sh $dir p2 dion-ab-p2-$arm-mem $P2_HOSTS --max-steps 5 \
            --memory-profiler-path $MEMORY/p2-$arm
    done
}

f8_probe() {
    $B/$H/run_cell.sh $B f8 dion-ab-f8-b-probe $F8_HOSTS --max-steps 3
}

f8() {
    for arm in a b; do
        dir=$A; [ $arm = b ] && dir=$B
        $dir/$H/run_cell.sh $dir f8 dion-ab-f8-$arm-1 $F8_HOSTS
    done
}

"$@"
