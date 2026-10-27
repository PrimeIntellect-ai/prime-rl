#!/bin/bash
# Serial main-vs-branch RL runs on job 7049. usage: rl_matrix.sh LAUNCHER CONFIG TAG [ARM...]
# LAUNCHER is rl_launch.sh (plain `uv run rl`) or rl_sbatch_launch.sh (generated single-node SLURM script).
set -uo pipefail
LAUNCHER=$1 CONFIG=$2 TAG=$3
shift 3
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(main branch)
L="bash /home/garrett/tmp/ngram-module/harness-e2e2/${LAUNCHER}"
declare -A ARM_ARGS=(
    [main]="/home/garrett/github/PrimeIntellect-ai/prime-rl-run-ngram-main-522bdc0 /tmp/garrett/ngram-e2e2-main main"
    [branch]="/home/garrett/github/PrimeIntellect-ai/prime-rl-run-ngram-branch-4766d44 /tmp/garrett/ngram-e2e2-branch branch"
)
for arm in "${ARMS[@]}"; do
    echo "START $arm $(date +%T)"
    $L ${ARM_ARGS[$arm]} $CONFIG ${TAG}-${arm} --inference.vllm.data-parallel-size 4
    echo "DONE $arm rc=$? $(date +%T)"
done
echo MATRIX_DONE
