#!/bin/bash
# Serial main-vs-branch export matrix on job 7049.
set -uo pipefail
L="bash /home/garrett/tmp/ngram-module/harness-e2e2/drivers_launch.sh"
M="/home/garrett/github/PrimeIntellect-ai/prime-rl-run-ngram-main-522bdc0 /tmp/garrett/ngram-e2e2-main main"
B="/home/garrett/github/PrimeIntellect-ai/prime-rl-run-ngram-branch-4766d44 /tmp/garrett/ngram-e2e2-branch branch"
CKPT=/home/garrett/tmp/ngram-module/ckpt/qwen38-synth-vl
G8=0,1,2,3,4,5,6,7
G2=0,1
P=29651
run() { echo "START $3 $(date +%T)"; $L "$@"; echo "DONE $3 rc=$? $(date +%T)"; }
run $M $G2 $P main-dp2-export-r0 export main dp2 --model.name $CKPT --model.dp-replicate 2
run $B $G2 $P branch-dp2-export-r0 export branch dp2 --model.name $CKPT --model.dp-replicate 2
run $M $G8 $P main-ep8-export-r0 export main ep8 --model.name $CKPT --model.ep 8
run $B $G8 $P branch-ep8-export-r0 export branch ep8 --model.name $CKPT --model.ep 8
echo MATRIX_DONE
