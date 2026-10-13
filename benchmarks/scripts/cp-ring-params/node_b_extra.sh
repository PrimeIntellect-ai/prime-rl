#!/bin/bash
# Node B extras on the #3925 layout: E3 (no AC offloading) on the branch, then an E0 rerun that py-spy dumps
# the trainer ranks if their logs go idle (the first E0 hung in a collective). This script must be an
# ancestor of the trainer processes for py-spy to attach under ptrace_scope=1.
# usage: node_b_extra.sh <e3925-worktree>
set -u
E3925=$1
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HOME/tmp/cp-ring-params
RING="--trainer.model.cp 2 --trainer.model.cp-style ring"
echo "$(date +%T) start e3-ring-3925-nooff"
"$HERE/run_rl_arm.sh" "$E3925" e3-ring-3925-nooff 0,1,2 10 $RING --trainer.model.ac-offloading None
echo "$(date +%T) end e3-ring-3925-nooff"
echo "$(date +%T) start e0-ring-3925-diag"
"$HERE/run_rl_arm.sh" "$E3925" e0-ring-3925-diag 0,1,2 10 $RING &
arm_pid=$!
dumped=0
while kill -0 "$arm_pid" 2>/dev/null; do
  sleep 60
  log=$(ls "$D"/runs/e0-ring-3925-diag/run/*/logs/attempt_1/trainer/torchrun/*/attempt_0/0/stderr.log 2>/dev/null | head -1)
  [ -z "$log" ] && continue
  idle=$(( $(date +%s) - $(stat -c %Y "$log") ))
  if [ "$idle" -gt 360 ] && [ "$dumped" -eq 0 ]; then
    for pid in $(pgrep -f "prime_rl.trainer.rl.train"); do
      py-spy dump --native --pid "$pid" > "$D/e0-diag-pyspy-$pid.txt" 2>&1
    done
    dumped=1
    echo "$(date +%T) dumped stacks after ${idle}s idle"
  fi
done
echo "$(date +%T) end e0-ring-3925-diag"
