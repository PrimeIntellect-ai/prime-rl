#!/bin/bash
# Node B, second numerics pass: the fake batch's random inference logprobs zero out importance-weighted RL
# losses, so these arms train on masked NLL (ce_loss_fn) to compare real gradients.
# usage: node_b_ce.sh <branch-worktree> <main-worktree>
set -u
BR=$1; MAIN=$2
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON="--model.moe-router-dtype bfloat16 --model.offload none --model.ac-offloading None --loss.type custom --loss.import-path prime_rl.trainer.rl.loss.ce_loss_fn"
RING="--model.cp 2 --model.cp-style ring"
ULY="--model.cp 2 --model.cp-style ulysses"
COMPILE="--model.compile"
tarm() { echo "$(date +%T) start $1"; "$HERE/run_trainer_arm.sh" "$2" "$1" "$3" "$4" "$5" "${@:6}"; echo "$(date +%T) end $1"; }
lane_ring() {
  tarm ce-br-ring-off      "$BR"   3,4 2 29711 $COMMON $RING
  tarm ce-main-ring-off    "$MAIN" 3,4 2 29711 $COMMON $RING
  tarm ce-br-ring-on       "$BR"   3,4 2 29711 $COMMON $RING $COMPILE
  tarm ce-main-ring-on     "$MAIN" 3,4 2 29711 $COMMON $RING $COMPILE
  tarm ce-main-ring-off-r2 "$MAIN" 3,4 2 29711 $COMMON $RING
}
lane_uly() {
  tarm ce-br-uly-off      "$BR"   5,6 2 29712 $COMMON $ULY
  tarm ce-main-uly-off    "$MAIN" 5,6 2 29712 $COMMON $ULY
  tarm ce-br-uly-on       "$BR"   5,6 2 29712 $COMMON $ULY $COMPILE
  tarm ce-main-uly-on     "$MAIN" 5,6 2 29712 $COMMON $ULY $COMPILE
  tarm ce-br-ring-off-r2  "$BR"   5,6 2 29712 $COMMON $RING
}
lane_cp1() {
  tarm ce-br-cp1-off    "$BR"   7 1 29713 $COMMON
  tarm ce-main-cp1-off  "$MAIN" 7 1 29713 $COMMON
  tarm ce-br-cp1-on     "$BR"   7 1 29713 $COMMON $COMPILE
  tarm ce-main-cp1-on   "$MAIN" 7 1 29713 $COMMON $COMPILE
}
lane_ring > "$HOME/tmp/cp-ring-params/lane_ce_ring.log" 2>&1 &
lane_uly > "$HOME/tmp/cp-ring-params/lane_ce_uly.log" 2>&1 &
lane_cp1 > "$HOME/tmp/cp-ring-params/lane_ce_cp1.log" 2>&1 &
wait
echo "$(date +%T) node B ce pass done"
