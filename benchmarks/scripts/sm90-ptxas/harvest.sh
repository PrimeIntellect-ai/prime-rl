#!/bin/bash
# usage: harvest.sh <gpu pair> <name> <cmd...>   runs cmd in the fix worktree with fresh caches and per-kernel benchmark harnesses
pair=$1; name=$2; shift 2
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-fix-cold-cache-ima
TB=$WT/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
d=/tmp/ptxas-scan-harvest-$name; rm -rf $d; mkdir -p $d/ind $d/tri $d/out
export CUDA_VISIBLE_DEVICES=$pair TORCHINDUCTOR_CACHE_DIR=$d/ind TRITON_CACHE_DIR=$d/tri TORCHINDUCTOR_BENCHMARK_KERNEL=1 TRITON_PTXAS_PATH=$TB/ptxas-blackwell OUT=$d/out
cd $WT
start=$(date +%s)
eval "$@" > /home/garrett/tmp/compile-tests-orch/ptxas-scan/logs/harvest-$name.log 2>&1
echo "harvest-$name RC=$? secs=$(($(date +%s)-start)) kernels=$(grep -rl 'def get_args' $d/ind | wc -l)" >> /home/garrett/tmp/compile-tests-orch/ptxas-scan/logs/harvest-summary.txt
