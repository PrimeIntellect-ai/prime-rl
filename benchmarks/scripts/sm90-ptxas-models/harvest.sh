#!/bin/bash
# usage: harvest.sh <gpu pair> <shape>
pair=$1; shape=$2
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-compile-test-resume
TB=$WT/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
M=/home/garrett/tmp/compile-tests-orch/ptxas-scan/models
d=/tmp/qkvm-harvest-$shape; rm -rf $d; mkdir -p $d/ind $d/tri $d/tl $d/out
export UV_NO_SYNC=1 SHAPE=$shape CUDA_VISIBLE_DEVICES=$pair TORCHINDUCTOR_CACHE_DIR=$d/ind TRITON_CACHE_DIR=$d/tri TILELANG_CACHE_DIR=$d/tl TORCHINDUCTOR_BENCHMARK_KERNEL=1 TRITON_PTXAS_PATH=$TB/ptxas-blackwell
cd $WT
start=$(date +%s)
uv run --no-sync torchrun --standalone --nproc-per-node=2 $M/modelshape.py $d/out > $M/harvest-$shape.log 2>&1
echo "harvest-$shape RC=$? secs=$(($(date +%s)-start)) kernels=$(grep -rl 'def get_args' $d/ind | wc -l)" >> $M/harvest-summary.txt
