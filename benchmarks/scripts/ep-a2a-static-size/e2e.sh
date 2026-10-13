#!/bin/bash
# End-to-end A/B on one exclusive 8-GPU node, every arm alone and serial. Per workload: main and branch with
# fullgraph off, main with fullgraph on (the ceiling), same-code repeats of both fullgraph-off arms, then 5-step
# profiler traces of both fullgraph-off arms.
# usage: e2e.sh <branch-worktree> <main-worktree> [workload...]
set -u
BR=$1; MAIN=$2; shift 2
WORKLOADS=${*:-qwen3-30b glm-air-8l}
HERE=$(cd "$(dirname "$0")" && pwd)
TRACES=${EP_A2A_TRACES:-$HOME/tmp/profiling/ep-a2a-static-size/traces}
FG="--model.compile.fullgraph"

arm() { echo "$(date +%T) start $2"; "$HERE/run_e2e_arm.sh" "$@"; echo "$(date +%T) end $2: $(tail -n 1 "${EP_A2A_E2E_OUT:-$HOME/tmp/ep-a2a-static-size/e2e}/$2/sft.log")"; }

for w in $WORKLOADS; do
  arm "$MAIN" "$w-main-off" "$w" 20
  arm "$BR" "$w-br-off" "$w" 20
  arm "$MAIN" "$w-main-fg" "$w" 20 $FG
  arm "$BR" "$w-br-off-r2" "$w" 20
  arm "$MAIN" "$w-main-off-r2" "$w" 20
  arm "$MAIN" "$w-trace-main-off" "$w" 5 --trace-path "$TRACES/$w-main-off"
  arm "$BR" "$w-trace-br-off" "$w" 5 --trace-path "$TRACES/$w-br-off"
done
echo "$(date +%T) e2e done"
