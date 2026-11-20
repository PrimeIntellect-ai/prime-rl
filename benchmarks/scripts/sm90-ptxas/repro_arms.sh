#!/bin/bash
WT=/home/garrett/github/PrimeIntellect-ai/prime-rl-fix-cold-cache-ima
TB=$WT/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin
S=/home/garrett/tmp/compile-tests-orch/ptxas-scan
cd $WT
for arm in p129 p133 noslp; do
  d=/tmp/ptxas-scan-repro-$arm; rm -rf $d; mkdir -p $d
  export TRITON_CACHE_DIR=$d
  case $arm in p133) export TRITON_PTXAS_PATH=$TB/ptxas-blackwell;; *) export TRITON_PTXAS_PATH=$TB/ptxas;; esac
  echo "== $arm $TRITON_PTXAS_PATH"
  if [ $arm = noslp ]; then uv run --no-sync python $S/noslp_run.py /home/garrett/tmp/compile-tests-orch/ima/ptxas_repro.py; else uv run --no-sync python /home/garrett/tmp/compile-tests-orch/ima/ptxas_repro.py; fi
  find $d -name '*.ptx' | while read p; do echo "$p packed16: $(grep -cE '\.(s|u|b)16x2|mov\.b32 *\{%rs' $p) v2.u16: $(grep -c 'v2\.\(u\|b\)16' $p)"; done
done
