#!/bin/bash
# Run the EP all-to-all A/B matrix on one 8-GPU node. Arms in a wave run concurrently on disjoint GPUs.
# All arms: bf16 router, ep=2, no offloading, TORCH_LOGS=recompiles,graph_breaks.
# usage: matrix.sh <branch-worktree> <main-worktree>
set -u
BR=$1; MAIN=$2
HERE=$(cd "$(dirname "$0")" && pwd)
NOOFF="--model.offload none --model.ac-offloading None"
TRAINER="--model.compile --model.moe-router-dtype bfloat16 --model.ep 2 $NOOFF --loss.type custom
  --loss.import-path prime_rl.trainer.rl.loss.ce_loss_fn"
SFT="--model.name samsja/mini-glm-moe --model.compile --model.ep 2 --max-steps 20 --data.seq-len 2048
  --data.batch-size 8 $NOOFF"
FG="--model.compile.fullgraph"
CP2="--model.cp 2"

trainer() { "$HERE/run_trainer_arm.sh" "$2" "$1" "$3" 2 "$4" ${@:5}; }
sft() { "$HERE/run_sft_arm.sh" "$2" "$1" "$3" "$4" ${@:5}; }
wave() { echo "$(date +%T) wave $1 start"; }

wave 1
trainer trainer-main-off "$MAIN" 0,1 29610 $TRAINER &
trainer trainer-br-off   "$BR"   2,3 29611 $TRAINER &
trainer trainer-main-fg  "$MAIN" 4,5 29612 $TRAINER $FG &
trainer trainer-br-fg    "$BR"   6,7 29613 $TRAINER $FG &
wait

wave 2
trainer trainer-main-off-r2 "$MAIN" 0,1 29620 $TRAINER &
trainer trainer-br-off-r2   "$BR"   2,3 29621 $TRAINER &
sft sft-main-off "$MAIN" 4,5 2 $SFT &
sft sft-br-off   "$BR"   6,7 2 $SFT &
wait

wave 3
sft sftcp-main-off "$MAIN" 0,1,2,3 4 $SFT $CP2 &
sft sftcp-br-off   "$BR"   4,5,6,7 4 $SFT $CP2 &
wait

wave 4
sft sftcp-main-fg "$MAIN" 0,1,2,3 4 $SFT $CP2 $FG &
sft sftcp-br-fg   "$BR"   4,5,6,7 4 $SFT $CP2 $FG &
wait
echo "$(date +%T) matrix done"
