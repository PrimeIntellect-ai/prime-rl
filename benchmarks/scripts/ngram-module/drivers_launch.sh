#!/bin/bash
# usage: drivers_launch.sh WORKTREE VENV ARM GPUS PORT RUN_NAME driver-args...
set -euo pipefail
WORKTREE=$1 VENV=$2 ARM=$3 GPUS=$4 PORT=$5 RUN_NAME=$6
shift 6
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DRIVER=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-ngram-module/benchmarks/scripts/ngram-module/export_driver.py
OUT_ROOT=/home/garrett/tmp/ngram-module/drivers/final-ab
LOG=${OUT_ROOT}/logs/${RUN_NAME}.log
CACHE=/tmp/garrett-drivers-cache/${RUN_NAME}
mkdir -p ${OUT_ROOT}/logs
srun --jobid=7002 --overlap -N1 -n1 --gres=gpu:8 bash -c "
export CUDA_VISIBLE_DEVICES=${GPUS} CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TRITON_CACHE_DIR=${CACHE}/triton TORCHINDUCTOR_CACHE_DIR=${CACHE}/inductor OMP_NUM_THREADS=1 VIRTUAL_ENV=${VENV}
rm -rf ${CACHE}; mkdir -p ${CACHE}
date +%s.%N; cd ${WORKTREE} && PRL_OUTPUT_DIR=${OUT_ROOT}/${ARM} uv run --active --no-sync torchrun \
  --nproc-per-node=${NPROC} --master-port=${PORT} ${DRIVER} --out-root ${OUT_ROOT} $*
rc=\$?; echo END \$(date +%s.%N) rc=\$rc; exit \$rc
" > "${LOG}" 2>&1
