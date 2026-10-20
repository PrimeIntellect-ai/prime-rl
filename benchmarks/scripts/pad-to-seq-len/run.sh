#!/usr/bin/env bash
# Usage: run.sh <rl|sft> <bf16|fp8> <true|false>
set -euo pipefail

kind=$1
precision=$2
pad=$3
name=pad-seq-$kind-$precision-pad-$pad

export PRL_OUTPUT_DIR=$HOME/prl_output_dir TORCH_LOGS=recompiles DG_JIT_PRINT_COMPILER_COMMAND=1
export DG_JIT_CACHE_DIR=/tmp/$USER/dg-$name
rm -rf "$DG_JIT_CACHE_DIR"

if [ "$kind" = rl ]; then
  model_prefix=--trainer.model
  pad_arg=(--orchestrator.pad-to-seq-len "$pad")
else
  model_prefix=--model
  pad_arg=(--data.pad-to-seq-len "$pad")
fi
fp8_args=()
if [ "$precision" = fp8 ]; then
  fp8_args=("$model_prefix.quantization.type" fp8 "$model_prefix.moe.compute.type" deepgemm_fp8)
fi
extra_args=()
if [ "$kind" = rl ] && [ "$precision" = fp8 ]; then
  extra_args=(--max-steps 12)
fi

uv run --no-sync "$kind" @ "benchmarks/scripts/pad-to-seq-len/$kind.toml" "${fp8_args[@]}" "${pad_arg[@]}" "${extra_args[@]}" \
  --run.name "$name" --monitors.wandb.name "$name" --no-dashboard

ls "$DG_JIT_CACHE_DIR/cache" | wc -l
