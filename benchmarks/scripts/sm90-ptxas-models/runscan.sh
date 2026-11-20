#!/bin/bash
# usage: runscan.sh <list> <outdir> <nshards> <gpu list csv> <arms...> ; dumpfile via DUMP env
list=$1; out=$2; n=$3; IFS=, read -ra gpus <<< "$4"; shift 4
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-compile-test-resume
TB=$WT/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
S=/home/garrett/tmp/compile-tests-orch/ptxas-scan
mkdir -p $out
cd $WT
for arm in "$@"; do
  for ((s=0; s<n; s++)); do
    c=$out/cache-$arm-$s; rm -rf $c; mkdir -p $c/ind $c/tri
    if [ $arm = p133 ]; then p=$TB/ptxas-blackwell; else p=$TB/ptxas; fi
    UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES=${gpus[$((s % ${#gpus[@]}))]} TRITON_PTXAS_PATH=$p TRITON_CACHE_DIR=$c/tri TORCHINDUCTOR_CACHE_DIR=$c/ind TORCHINDUCTOR_COMPILE_THREADS=1 \
      uv run --no-sync python $S/diffscan.py drive $arm $list $out $s $n $DUMP > $out/drive-$arm-$s.log 2>&1 &
  done
done
wait
