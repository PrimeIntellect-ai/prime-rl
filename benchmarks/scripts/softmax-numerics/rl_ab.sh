#!/bin/bash
# One arm of the 25-step RL comparison: run from the checkout under test on one 8-GPU node.
#   bash benchmarks/scripts/softmax-numerics/rl_ab.sh <arm name> <context parallel size>
# Credentials (W&B, the math env's judge) and PRL_OUTPUT_DIR come from the environment.
set -e
ARM=$1
CP=$2
shift 2
NAME=hendrycks-qwen3-1.7b-25step-${ARM}-cp${CP}
export TORCH_LOGS=recompiles
uv run --no-sync rl @ examples/basic/hendrycks-sanity/rl.toml \
  --max-steps 25 \
  --no-orchestrator.eval \
  --inference.vllm.data-parallel-size 4 \
  --trainer.model.cp "$CP" \
  --run.name "$NAME" \
  --monitors.wandb.name "$NAME" \
  --no-dashboard \
  "$@"
