#!/bin/bash
# Serial main-vs-final driver matrix on job 7002.
set -uo pipefail
L="bash /home/garrett/github/PrimeIntellect-ai/prime-rl-feat-ngram-module/benchmarks/scripts/ngram-module/drivers_launch.sh"
M="/home/garrett/github/PrimeIntellect-ai/prime-rl-run-ngram-main-1810d5b /tmp/garrett/ngram-main-1810d5b main"
F="/home/garrett/github/PrimeIntellect-ai/prime-rl-run-ngram-branch-6305757 /tmp/garrett/ngram-branch-6305757 final"
G8=0,1,2,3,4,5,6,7
G2=0,1
P=29651
run() { echo "START $3 $(date +%T)"; $L "$@"; echo "DONE $3 rc=$? $(date +%T)"; }
run $M $G2 $P main-dp2-export-r0 export main dp2 --model.dp-replicate 2
run $M $G8 $P main-ep8-export-r0 export main ep8 --model.ep 8
run $F $G8 $P final-ep8-export-r0 export final ep8 --model.ep 8
run $M $G8 $P main-ep8-grads-r0 --repeat 0 grads main ep8 --model.ep 8
run $M $G8 $P main-ep8-grads-r1 --repeat 1 grads main ep8 --model.ep 8
run $F $G8 $P final-ep8-grads-r0 --repeat 0 grads final ep8 --model.ep 8
run $F $G8 $P final-ep8-grads-r1 --repeat 1 grads final ep8 --model.ep 8
run $M $G2 $P main-dp2-grads-r0 --repeat 0 --optimizer-step grads main dp2 --model.dp-replicate 2
run $M $G2 $P main-dp2-grads-r1 --repeat 1 --optimizer-step grads main dp2 --model.dp-replicate 2
run $F $G2 $P final-dp2-grads-r0 --repeat 0 --optimizer-step grads final dp2 --model.dp-replicate 2
run $M $G8 $P main-dp2x4-grads-r0 --repeat 0 --optimizer-step grads main dp2x4 --model.dp-replicate 2
run $M $G8 $P main-dp2x4-grads-r1 --repeat 1 --optimizer-step grads main dp2x4 --model.dp-replicate 2
run $F $G8 $P final-dp2x4-grads-r0 --repeat 0 --optimizer-step grads final dp2x4 --model.dp-replicate 2
echo MATRIX_DONE
