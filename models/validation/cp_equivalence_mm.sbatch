#!/bin/bash
#SBATCH --job-name=cp-equiv-mm
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/hubert/nemotron-vl/prime-rl/models/validation/logs/cp_equiv_mm_%j.log

set -euo pipefail
cd /home/hubert/nemotron-vl/prime-rl

export HF_HOME=/tmp/hubert-hf
export TMPDIR=/tmp/hubert-tmp
export TRITON_CACHE_DIR=/tmp/hubert-triton-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR"

SCRATCH=$(mktemp -d "$TMPDIR/cp_equiv.XXXX")

uv sync --all-extras --all-packages --group mamba-ssm

echo "=== cp=1 reference ==="
uv run --no-sync torchrun --standalone --nproc-per-node 1 \
    models/validation/cp_equivalence_mm.py --cp-size 1 --scratch "$SCRATCH"

echo "=== cp=8 comparison ==="
uv run --no-sync torchrun --standalone --nproc-per-node 8 \
    models/validation/cp_equivalence_mm.py --cp-size 8 --scratch "$SCRATCH"

rm -rf "$SCRATCH"
