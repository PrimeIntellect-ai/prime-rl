#!/usr/bin/env bash
# Usage: launch.sh <variant> [extra sft args]   e.g. launch.sh branch-bf16-fg --dry-run
set -euo pipefail

AB_DIR=$(cd "$(dirname "$0")" && pwd)
BRANCH_WORKTREE=$HOME/github/PrimeIntellect-ai/prime-rl-fixfa2-fa3-custom-op
MAIN_WORKTREE=$HOME/github/PrimeIntellect-ai/prime-rl-main
RUNS_DIR=$HOME/tmp/fa-custom-op/runs

# variant -> "code router_dtype fullgraph attn"
declare -A VARIANT=(
  [main-bf16]="main bfloat16 false flash_attention_3"
  [main-bf16-r2]="main bfloat16 false flash_attention_3"
  [branch-bf16]="branch bfloat16 false flash_attention_3"
  [branch-bf16-r2]="branch bfloat16 false flash_attention_3"
  [branch-bf16-fg]="branch bfloat16 true flash_attention_3"
  [main-fp32]="main float32 false flash_attention_3"
  [branch-fp32]="branch float32 false flash_attention_3"
  [branch-bf16-fg-fa2]="branch bfloat16 true flash_attention_2"
)

variant=$1
shift
read -r code router fullgraph attn <<<"${VARIANT[$variant]}"

worktree=$BRANCH_WORKTREE
if [ "$code" = "main" ]; then
  worktree=$MAIN_WORKTREE
fi

compile_args=()
if [ "$fullgraph" = "true" ]; then
  compile_args=(--model.compile.fullgraph)
fi

cd "$worktree"
env -u HF_HOME PRL_OUTPUT_DIR="$RUNS_DIR" uv run --no-sync sft @ "$AB_DIR/sft-base.toml" \
  --run.name "$variant" \
  --model.ep 8 --model.moe-router-dtype "$router" --model.attn "$attn" \
  "${compile_args[@]}" \
  --monitors.wandb.name "$variant" \
  --monitors.wandb.tags "[\"$code\",\"router-$router\",\"fullgraph-$fullgraph\",\"$attn\"]" \
  --no-dashboard "$@"
