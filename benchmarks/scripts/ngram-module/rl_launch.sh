#!/bin/bash
# usage: rl_launch.sh WORKTREE VENV ARM CONFIG RUN_NAME rl-args...
set -euo pipefail
WORKTREE=$1 VENV=$2 ARM=$3 CONFIG=$4 RUN_NAME=$5
shift 5
OUT_ROOT=/home/garrett/tmp/ngram-module/e2e2/node-b/rl
LOG=${OUT_ROOT}/logs/${RUN_NAME}.log
CACHE=/tmp/garrett/ngram-e2e2-cache/${RUN_NAME}
mkdir -p ${OUT_ROOT}/logs
srun --jobid=7049 --overlap -N1 -n1 --gres=gpu:8 bash -c "
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=/home/huggingface VIRTUAL_ENV=${VENV} \
  TRITON_CACHE_DIR=${CACHE}/triton TORCHINDUCTOR_CACHE_DIR=${CACHE}/inductor VLLM_CACHE_ROOT=${CACHE}/vllm \
  FLASHINFER_WORKSPACE_BASE=${CACHE}/flashinfer DG_JIT_CACHE_DIR=${CACHE}/deep_gemm
rm -rf ${CACHE}; mkdir -p ${CACHE}
date +%s.%N; cd ${WORKTREE} && PRL_OUTPUT_DIR=${OUT_ROOT}/${ARM} uv run --active --no-sync rl @ ${CONFIG} \
  --output-dir ${OUT_ROOT}/${ARM} --run.name ${RUN_NAME} \
  --trainer.model.conversion-dir ${OUT_ROOT}/conv/${ARM} $*
rc=\$?; echo END \$(date +%s.%N) rc=\$rc; exit \$rc
" > "${LOG}" 2>&1
