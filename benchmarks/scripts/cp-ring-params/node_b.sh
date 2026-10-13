#!/bin/bash
# Node B: smoke runs on GPUs 0-2 (serial), numerics trainer arms on GPUs 3-7.
# usage: node_b.sh <branch-worktree> <main-worktree>
set -u
BR=$1; MAIN=$2
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON="--model.moe-router-dtype bfloat16 --model.offload none --model.ac-offloading None"
RING="--model.cp 2 --model.cp-style ring"
ULY="--model.cp 2 --model.cp-style ulysses"
COMPILE="--model.compile"
tarm() { echo "$(date +%T) start $1"; "$HERE/run_trainer_arm.sh" "$2" "$1" "$3" "$4" "$5" "${@:6}"; echo "$(date +%T) end $1"; }
lane_ring() {
  tarm num-br-ring-off     "$BR"   3,4 2 29701 $COMMON $RING
  tarm num-main-ring-off   "$MAIN" 3,4 2 29701 $COMMON $RING
  tarm num-br-ring-on      "$BR"   3,4 2 29701 $COMMON $RING $COMPILE
  tarm num-main-ring-on    "$MAIN" 3,4 2 29701 $COMMON $RING $COMPILE
  tarm num-main-ring-off-r2 "$MAIN" 3,4 2 29701 $COMMON $RING
}
lane_uly() {
  tarm num-br-uly-off    "$BR"   5,6 2 29702 $COMMON $ULY
  tarm num-main-uly-off  "$MAIN" 5,6 2 29702 $COMMON $ULY
  tarm num-br-uly-on     "$BR"   5,6 2 29702 $COMMON $ULY $COMPILE
  tarm num-main-uly-on   "$MAIN" 5,6 2 29702 $COMMON $ULY $COMPILE
  tarm num-br-ring-off-r2 "$BR"  5,6 2 29702 $COMMON $RING
}
lane_cp1() {
  tarm num-br-cp1-off    "$BR"   7 1 29703 $COMMON
  tarm num-main-cp1-off  "$MAIN" 7 1 29703 $COMMON
  tarm num-br-cp1-on     "$BR"   7 1 29703 $COMMON $COMPILE
  tarm num-main-cp1-on   "$MAIN" 7 1 29703 $COMMON $COMPILE
}
lane_ring > "$HOME/tmp/cp-ring-params/lane_ring.log" 2>&1 &
lane_uly > "$HOME/tmp/cp-ring-params/lane_uly.log" 2>&1 &
lane_cp1 > "$HOME/tmp/cp-ring-params/lane_cp1.log" 2>&1 &
arm() { echo "$(date +%T) start $1"; "$HERE/run_rl_arm.sh" "$2" "$1" 0,1,2 "${@:3}"; echo "$(date +%T) end $1"; }
echo "$(date +%T) start sft-ring"; "$HERE/run_sft_arm.sh" "$BR" sft-ring 0,1 2 --model.cp 2 --model.cp-style ring --model.compile --max-steps 10; echo "$(date +%T) end sft-ring"
# VLMs only allow ulysses CP (config validator), which takes the deferred CP path in the model.
CP_RING_RL_CONFIG=configs/ci/nightly/multimodal_color_codeword.toml arm vlm-uly "$BR" 5 \
  --trainer.model.cp 2 --trainer.model.cp-style ulysses --orchestrator.batch-size 64 --inference.vllm.max-model-len 4096
wait
echo "$(date +%T) node B done"
