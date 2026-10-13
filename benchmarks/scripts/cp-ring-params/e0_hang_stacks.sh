#!/bin/bash
# Rerun E0 (#3925 layout, ring cp2, AC offloading on) and, once the trainer logs go idle, SIGABRT the trainer
# ranks so faulthandler prints every thread's Python stack into the rank stderr logs.
# usage: e0_hang_stacks.sh <e3925-worktree> <name>
set -u
E3925=$1; NAME=$2
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HOME/tmp/cp-ring-params
export PYTHONFAULTHANDLER=1
echo "$(date +%T) start $NAME"
"$HERE/run_rl_arm.sh" "$E3925" "$NAME" 0,1,2 10 --trainer.model.cp 2 --trainer.model.cp-style ring &
arm_pid=$!
signalled=0
while kill -0 "$arm_pid" 2>/dev/null; do
  sleep 30
  log=$(ls "$D"/runs/"$NAME"/run/*/logs/attempt_1/trainer/torchrun/*/attempt_0/0/stderr.log 2>/dev/null | head -1)
  [ -z "$log" ] && continue
  idle=$(( $(date +%s) - $(stat -c %Y "$log") ))
  if [ "$idle" -gt 360 ] && [ "$signalled" -eq 0 ]; then
    torchrun_pid=$(pgrep -f "role=trainer" | head -1)
    ranks=$(pgrep -P "$torchrun_pid")
    echo "$(date +%T) idle ${idle}s, torchrun $torchrun_pid, SIGABRT ranks: $ranks"
    kill -ABRT $ranks
    signalled=1
  fi
done
echo "$(date +%T) end $NAME"
