#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/matej/dev/prime-rl/scripts/glm5_weight_transfer/test_fp8.out

cd /home/matej/dev/prime-rl
export VLLM_USE_DEEP_GEMM=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="/home/matej/dev/prime-rl/scripts/glm5_weight_transfer:$PYTHONPATH"

uv run --env-file /dev/null env PYTHONPATH="/home/matej/dev/prime-rl/scripts/glm5_weight_transfer:$PYTHONPATH" python scripts/glm5_weight_transfer/test_weight_transfer.py 2>&1
