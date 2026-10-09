#!/bin/bash
# One untraced 15-step run of one arm on the current node, fresh caches. usage: run_arm.sh ARM REP (ARM: bf16, before, after)
set -u
arm=$1 rep=$2
I=$HOME/tmp/profiling/fp8-cleanup-h200
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-fp8-cleanup
C=$WT/benchmarks/scripts/fp8_cleanup_h200_refresh
cd $WT
export PRL_OUTPUT_DIR=/home/garrett/prl_output_dir HF_HOME=/home/garrett/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
name=fp8refresh-h200-$arm-r$rep
pp=""; [ "$arm" = before ] && pp=$I/arms/main-88faa6dc2/src
overlay=(@ $C/sft_fp8.toml); [ "$arm" = bf16 ] && overlay=()
rm -rf /tmp/$USER/$name; mkdir -p /tmp/$USER/$name
echo "=== $name $(hostname) start $(date -u +%T)"
PYTHONPATH=$pp TRITON_CACHE_DIR=/tmp/$USER/$name/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/$name/inductor \
  DG_JIT_CACHE_DIR=/tmp/$USER/$name/dg TILELANG_CACHE_DIR=/tmp/$USER/$name/tilelang \
  timeout 1500 uv run --no-sync sft @ $C/sft_1n_32k.toml "${overlay[@]}" --run.name $name --clean > $C/results/$name.log 2>&1
echo "=== $name exit $? $(date -u +%T)"
