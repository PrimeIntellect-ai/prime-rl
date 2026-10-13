#!/bin/bash
# CP=2 SFT arms with fullgraph off on the CP attention-params fix, alone and with this branch's fix on top.
# The combo arm starts immediately on GPUs 0-3; the cp-only arm waits for the reverse-text RL arms to free GPUs 4-7.
# usage: cp_combo.sh <cp-only-worktree> <combo-worktree> <traces-and-rl-log>
set -u
CPONLY=$1; COMBO=$2; RL_LOG=$3
HERE=$(cd "$(dirname "$0")" && pwd)
SFT="--model.name samsja/mini-glm-moe --model.compile --model.ep 2 --max-steps 20 --data.seq-len 2048
  --data.batch-size 8 --model.offload none --model.ac-offloading None --model.cp 2"

"$HERE/run_sft_arm.sh" "$COMBO" sftcp-combo-off 0,1,2,3 4 $SFT &
until grep -q "traces and rl done" "$RL_LOG" 2>/dev/null; do sleep 10; done
"$HERE/run_sft_arm.sh" "$CPONLY" sftcp-cponly-off 4,5,6,7 4 $SFT &
wait
echo "$(date +%T) cp combo done"
