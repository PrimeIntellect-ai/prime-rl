#!/bin/bash
# Submit one exclusive single-node job per run. Usage: submit_1n.sh CONFIG ARM KIND REP
C=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-fp8-cleanup/benchmarks/scripts/fp8_cleanup_h200_table
sbatch --parsable --exclusive -N1 --gres=gpu:8 -p all --time=00:40:00 -J fp8table-1n \
  -o $C/results/slurm-%j.log --wrap "bash $C/run_1n.sh $*"
