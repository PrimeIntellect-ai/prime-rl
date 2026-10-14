#!/bin/bash
# Kernel smoke check for the MXFP8 EP all-to-all arms, run inside the node allocation.
# usage: srun --jobid <id> --overlap -N1 smoke.sh <worktree>
set -u
cd "$1"
git rev-parse HEAD
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
uv sync --all-extras --all-packages --check
uv run --no-sync python -c "import prime_kernels, importlib.metadata as m; print(m.version('prime-kernels')); print(prime_kernels.status())"
uv run --no-sync python -c "import prime_kernels; prime_kernels.load('mxfp8_moe'); print('mxfp8_moe ok')"
