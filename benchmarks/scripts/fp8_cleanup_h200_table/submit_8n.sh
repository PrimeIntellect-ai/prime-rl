#!/bin/bash
# Submit one 8-node run through the multi_node launcher (its [slurm] section sizes and time-limits the job). Usage: submit_8n.sh ARM
arm=$1 name=fp8table-8n-$1
source /home/garrett/github/PrimeIntellect-ai/prime-rl-feat-fp8-cleanup/benchmarks/scripts/fp8_cleanup_h200_table/env.sh $arm $name
cd $WT
uv run --no-sync sft @ $C/sft_8n.toml "${OVERLAY[@]}" --run.name $name --clean
