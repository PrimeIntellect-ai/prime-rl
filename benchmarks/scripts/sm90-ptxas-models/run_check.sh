#!/bin/bash
gpu=$1; arm=$2
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-compile-test-resume
TB=$WT/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
V=/home/garrett/tmp/compile-tests-orch/taskB-verify
if [ $arm = p133 ]; then p=$TB/ptxas-blackwell; else p=$TB/ptxas; fi
c=/tmp/vB-check-$arm; rm -rf $c; mkdir -p $c/ind $c/tri $c/src
$p --version | tail -1
files=$(grep -L 'async_compile.triton(' $(cat $V/catfiles-glm05b.txt $V/catfiles-nemo32.txt) | xargs grep -l 'def triton_poi_fused_[_a-z]*cat[a-z_]*select_backward')
cd $WT
CUDA_VISIBLE_DEVICES=$gpu TRITON_PTXAS_PATH=$p TRITON_CACHE_DIR=$c/tri TORCHINDUCTOR_CACHE_DIR=$c/ind uv run --no-sync python $V/check.py $arm $c/src $files
echo "PTXVERSION $arm $(find $c/tri -name '*.ptx' | xargs grep -h '^\.version' | sort | uniq -c | tr '\n' ' ') nptx=$(find $c/tri -name '*.ptx' | wc -l)"
