#!/bin/bash

#SBATCH --job-name=gsm8k-recheck-wrong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=all
#SBATCH --output=outputs-wrong/job_%j.log
#SBATCH --error=outputs-wrong/job_%j.log

# Exit immediately if any setup command or the launcher fails so the SLURM job
# releases its allocation instead of holding nodes after a crash.
set -e

export PROJECT_DIR=/home/fares/prime-rl-resume

# Setup environment
cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate
uv sync --all-extras


uv run rl @ outputs-wrong/configs/rl.toml
