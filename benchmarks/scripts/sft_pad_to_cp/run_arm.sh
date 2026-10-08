#!/usr/bin/env bash
# Usage: run_arm.sh <arm-name> <node> <run-name> <config> [extra sft args]. Runs single-node sft on a held node.
set -euo pipefail
arm=$HOME/tmp/sft_pad_to_cp/arms/$1; node=$2; run=$3; config=$4
runs=$HOME/tmp/sft_pad_to_cp/runs
mkdir -p "$runs"
srun --jobid=3409 --overlap -N1 --ntasks-per-node=1 --nodelist="$node" env -u HF_HOME bash -c '
  set -euo pipefail
  cache=/tmp/garrett/sftpad/'"$run"'
  rm -rf "$cache" && mkdir -p "$cache"
  export HF_HUB_CACHE=/home/huggingface/hub TRITON_CACHE_DIR=$cache/triton TORCHINDUCTOR_CACHE_DIR=$cache/inductor
  export TRITON_PRINT_AUTOTUNING=1 TORCH_LOGS=recompiles PRL_OUTPUT_DIR='"$runs"'
  cd '"$arm"'
  uv run --no-sync sft @ '"$config"' --run.name '"$run"' --monitors.wandb.name '"$run"' --no-dashboard '"${*:5}"'
'
