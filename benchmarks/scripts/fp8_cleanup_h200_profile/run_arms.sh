#!/bin/bash
# Serial A/B on one 8x H200 node: traced (5 steps) and untraced (15 steps) per arm, fresh caches per run.
set -u
I=$HOME/tmp/profiling/fp8-cleanup-h200
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-fp8-cleanup
cd $WT
export PRL_OUTPUT_DIR=/home/garrett/prl_output_dir HF_HOME=/home/garrett/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
run() {
  local arm=$1 kind=$2; local name=fp8cleanup-h200-$arm-$kind
  local extra=(); [ "$kind" = trace ] && extra=(--max-steps 5 --trace-path $I/traces/$arm)
  local pp=""; [ "$arm" = before ] && pp=$I/arms/main-88faa6dc2/src
  local overlay=(@ $I/configs/sft_fp8.toml); [ "$arm" = bf16 ] && overlay=()
  rm -rf /tmp/$USER/$name; mkdir -p /tmp/$USER/$name
  echo "=== $name start $(date -u +%T)"
  PYTHONPATH=$pp TRITON_CACHE_DIR=/tmp/$USER/$name/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/$name/inductor \
    DG_JIT_CACHE_DIR=/tmp/$USER/$name/dg TILELANG_CACHE_DIR=$HOME/tmp/profiling/caches/$name/tilelang \
    timeout 1800 uv run --no-sync sft @ $I/configs/sft_1n_32k.toml "${overlay[@]}" \
    --run.name $name --clean "${extra[@]}" > $I/$name.log 2>&1
  echo "=== $name exit $? $(date -u +%T)"
}
if [ $# -gt 0 ]; then for spec in "$@"; do run ${spec%%:*} ${spec#*:}; done; else
run before trace; run after trace; run before time; run after time; fi
