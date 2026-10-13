#!/bin/bash
# CP=2 SFT arms with fullgraph off on the CP attention-params fix, alone and with this branch's fix on top.
# usage: cp_combo.sh <cp-only-worktree> <combo-worktree>
set -u
CPONLY=$1; COMBO=$2
HERE=$(cd "$(dirname "$0")" && pwd)
SFT="--model.name samsja/mini-glm-moe --model.compile --model.ep 2 --max-steps 20 --data.seq-len 2048
  --data.batch-size 8 --model.offload none --model.ac-offloading None --model.cp 2"

"$HERE/run_sft_arm.sh" "$CPONLY" sftcp-cponly-off 0,1,2,3 4 $SFT &
"$HERE/run_sft_arm.sh" "$COMBO" sftcp-combo-off 4,5,6,7 4 $SFT &
wait
echo "$(date +%T) cp combo done"
