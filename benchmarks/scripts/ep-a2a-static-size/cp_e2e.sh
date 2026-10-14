#!/bin/bash
# GLM-4.5-Air (8 layers, real SFT data) with cp=2 and ep=4 on one 8-GPU node, fullgraph off, every arm alone and
# serial. The four trees share one base: base, base + this fix, base + #4068 (CP attention params as forward
# args), and base + both. The last two get same-code repeats.
# usage: cp_e2e.sh <base-wt> <fix-wt> <4068-wt> <both-wt>
set -u
BASE=$1; FIX=$2; CP4068=$3; BOTH=$4
HERE=$(cd "$(dirname "$0")" && pwd)
export EP_A2A_EP=4
OUT=${EP_A2A_E2E_OUT:-$HOME/tmp/ep-a2a-static-size/e2e}

arm() { echo "$(date +%T) start $2"; "$HERE/run_e2e_arm.sh" "$1" "$2" glm-air-8l 20 --model.cp 2; echo "$(date +%T) end $2: $(tail -n 1 "$OUT/$2/sft.log")"; }

arm "$BASE" cp2-base
arm "$CP4068" cp2-4068
arm "$BOTH" cp2-both
arm "$FIX" cp2-fix
arm "$BOTH" cp2-both-r2
arm "$CP4068" cp2-4068-r2
echo "$(date +%T) cp e2e done"
