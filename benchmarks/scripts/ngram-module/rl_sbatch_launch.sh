#!/bin/bash
# Generate the single-node SLURM script for an RL run, then run it inside job 7049 with the node-local venv.
# For NIXL, ModelExpress state moves out of the run directory: the template's default location makes the launched
# `rl` refuse the run directory as already used.
# usage: rl_sbatch_launch.sh WORKTREE VENV ARM CONFIG RUN_NAME rl-args...
set -euo pipefail
WORKTREE=$1 VENV=$2 ARM=$3 CONFIG=$4 RUN_NAME=$5
shift 5
OUT_ROOT=/home/garrett/tmp/ngram-module/e2e2/node-b/rl
RUN_DIR=${OUT_ROOT}/${ARM}/${RUN_NAME}
LOG=${OUT_ROOT}/logs/${RUN_NAME}.log
CACHE=/tmp/garrett/ngram-e2e2-cache/${RUN_NAME}
mkdir -p ${OUT_ROOT}/logs
srun --jobid=7049 --overlap -N1 -n1 --gres=gpu:8 bash -c "
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME=/home/huggingface VIRTUAL_ENV=${VENV} PRL_OUTPUT_DIR=${OUT_ROOT}/${ARM} \
  TRITON_CACHE_DIR=${CACHE}/triton TORCHINDUCTOR_CACHE_DIR=${CACHE}/inductor VLLM_CACHE_ROOT=${CACHE}/vllm \
  FLASHINFER_WORKSPACE_BASE=${CACHE}/flashinfer DG_JIT_CACHE_DIR=${CACHE}/deep_gemm
rm -rf ${CACHE}; mkdir -p ${CACHE}
date +%s.%N; cd ${WORKTREE}
uv run --active --no-sync rl @ ${CONFIG} --output-dir ${OUT_ROOT}/${ARM} --run.name ${RUN_NAME} \
  --trainer.model.conversion-dir ${OUT_ROOT}/conv/${ARM} --slurm.project-dir ${WORKTREE} --slurm.job-name ${RUN_NAME} \
  --dry-run $*
SCRIPT=${RUN_DIR}/launcher/rl.sbatch
grep -qx 'source .venv/bin/activate' \$SCRIPT && grep -qx 'uv sync --all-extras --all-packages' \$SCRIPT \
  && grep -q '^uv run rl @' \$SCRIPT
sed -e 's|^source .venv/bin/activate\$|export VIRTUAL_ENV=${VENV}|' -e '/^uv sync --all-extras --all-packages\$/d' \
  -e 's|^uv run rl @|uv run --active --no-sync rl @|' -e 's|^MX_STATE_DIR=.*|MX_STATE_DIR=${CACHE}/modelexpress|' \
  \$SCRIPT > ${RUN_DIR}/launcher/rl.node-local.sh
bash ${RUN_DIR}/launcher/rl.node-local.sh
rc=\$?; echo END \$(date +%s.%N) rc=\$rc; exit \$rc
" > "${LOG}" 2>&1
