#!/bin/bash
#SBATCH --job-name=fp8-blockwise-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --nodelist=qpn01gpu008
#SBATCH --partition=all
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/home/outputs/int4-debug/fp8-blockwise/job_%j.log
#SBATCH --error=/home/outputs/int4-debug/fp8-blockwise/job_%j.log
set -euo pipefail
cd /home/prod/fp8-blockwise-investigation
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export HF_HUB_CACHE=/home/hf-cache/hub
export HF_XET_CACHE=/home/hf-cache/xet
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export TMPDIR=/tmp/prod-fp8-probe-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=${TMPDIR}/triton
export VLLM_CACHE_ROOT=${TMPDIR}/vllm
export FLASHINFER_WORKSPACE_BASE=${FP8_PROBE_FLASHINFER_CACHE:-${TMPDIR}/flashinfer}
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used --format=csv
uv run --no-sync python -c 'import torch, vllm; print("VERSIONS", torch.__version__, vllm.__version__); assert torch.cuda.device_count() == 8'
uv run --no-sync python tools/fp8_inference_probe.py "$@"
