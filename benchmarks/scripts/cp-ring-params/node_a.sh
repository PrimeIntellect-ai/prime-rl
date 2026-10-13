#!/bin/bash
# Node A: recompile and timing arms, serial and alone on the node. Branch and main interleaved.
# usage: node_a.sh <branch-worktree> <main-worktree>
set -u
BR=$1; MAIN=$2
HERE=$(cd "$(dirname "$0")" && pwd)
NOOFF="--trainer.model.offload none --trainer.model.ac-offloading None"
BF16="--trainer.model.moe-router-dtype bfloat16"
RING="--trainer.model.cp 2 --trainer.model.cp-style ring"
ULY="--trainer.model.cp 2 --trainer.model.cp-style ulysses"
CP1="--trainer.model.cp 1"
arm() { echo "$(date +%T) start $1"; "$HERE/run_rl_arm.sh" "$2" "$1" 0,1,2 20 "${@:3}"; echo "$(date +%T) end $1"; }
arm br-bf16-ring   "$BR"   $BF16 $RING $NOOFF
arm main-bf16-ring "$MAIN" $BF16 $RING $NOOFF
arm br-bf16-uly    "$BR"   $BF16 $ULY $NOOFF
arm main-bf16-uly  "$MAIN" $BF16 $ULY $NOOFF
arm br-bf16-cp1    "$BR"   $BF16 $CP1 $NOOFF
arm main-bf16-cp1  "$MAIN" $BF16 $CP1 $NOOFF
arm br-fp32-ring   "$BR"   $RING $NOOFF
arm main-fp32-ring "$MAIN" $RING $NOOFF
arm br-fp32-uly    "$BR"   $ULY $NOOFF
arm main-fp32-uly  "$MAIN" $ULY $NOOFF
arm br-bf16-ring-r2   "$BR"   $BF16 $RING $NOOFF
arm main-bf16-ring-r2 "$MAIN" $BF16 $RING $NOOFF
echo "$(date +%T) node A done"
