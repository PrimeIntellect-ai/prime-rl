#!/bin/bash
# The untraced arms of e2e.sh for the given workloads: parent and branch with fullgraph off, parent with fullgraph
# on, and same-code repeats of both fullgraph-off arms.
# usage: e2e_untraced.sh <branch-worktree> <parent-worktree> <workload...>
set -u
BR=$1; MAIN=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
FG="--model.compile.fullgraph"

arm() { echo "$(date +%T) start $2"; "$HERE/run_e2e_arm.sh" "$@"; echo "$(date +%T) end $2: $(tail -n 1 "${EP_A2A_E2E_OUT:-$HOME/tmp/ep-a2a-static-size/e2e}/$2/sft.log")"; }

for w in "$@"; do
  arm "$MAIN" "$w-main-off" "$w" 20
  arm "$BR" "$w-br-off" "$w" 20
  arm "$MAIN" "$w-main-fg" "$w" 20 $FG
  arm "$BR" "$w-br-off-r2" "$w" 20
  arm "$MAIN" "$w-main-off-r2" "$w" 20
done
echo "$(date +%T) e2e untraced done"
