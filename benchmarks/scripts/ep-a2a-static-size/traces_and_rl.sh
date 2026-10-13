#!/bin/bash
# Profiler traces (SFT, 5 steps) then fullgraph-on trainer repeats on GPUs 0-3, alongside reverse-text-moe RL
# arms run serially on GPUs 4-6.
# All arms: bf16 router, ep=2, no offloading, compile on with fullgraph off.
# usage: traces_and_rl.sh <branch-worktree> <main-worktree>
set -u
BR=$1; MAIN=$2
HERE=$(cd "$(dirname "$0")" && pwd)
TRACES=${EP_A2A_TRACES:-$HOME/tmp/profiling/ep-a2a-static-size/traces}
SFT="--model.name samsja/mini-glm-moe --model.compile --model.ep 2 --max-steps 5 --data.seq-len 2048
  --data.batch-size 8 --model.offload none --model.ac-offloading None"
RL="--trainer.model.compile --trainer.model.moe-router-dtype bfloat16 --trainer.model.ep 2
  --trainer.model.offload none --trainer.model.ac-offloading None"
TRAINER="--model.compile --model.moe-router-dtype bfloat16 --model.ep 2 --model.offload none
  --model.ac-offloading None --loss.type custom --loss.import-path prime_rl.trainer.rl.loss.ce_loss_fn"
FG="--model.compile.fullgraph"

(
  "$HERE/run_sft_arm.sh" "$MAIN" trace-main-off 0,1 2 $SFT --trace-path "$TRACES/main-off" &
  "$HERE/run_sft_arm.sh" "$BR" trace-br-off 2,3 2 $SFT --trace-path "$TRACES/br-off" &
  wait
  "$HERE/run_trainer_arm.sh" "$MAIN" trainer-main-fg-r2 0,1 2 29630 $TRAINER $FG &
  "$HERE/run_trainer_arm.sh" "$BR" trainer-br-fg-r2 2,3 2 29631 $TRAINER $FG &
  wait
) &
"$HERE/run_rl_arm.sh" "$MAIN" rl-main-off 4,5,6 20 $RL
"$HERE/run_rl_arm.sh" "$BR" rl-br-off 4,5,6 20 $RL
wait
echo "$(date +%T) traces and rl done"
