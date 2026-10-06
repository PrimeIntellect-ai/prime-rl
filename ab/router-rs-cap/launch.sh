#!/usr/bin/env bash
# Usage: [SCENARIO=<scenario>] launch.sh <fsdp16|ep8> <variant> [--trace] [extra sft args]
#   e.g. SCENARIO=s4k-noac launch.sh ep8 fp32-cap2 --dry-run   (no SCENARIO: the 16k, full-AC baseline)
set -euo pipefail

AB_DIR=$(cd "$(dirname "$0")" && pwd)
BRANCH_WORKTREE=$HOME/github/PrimeIntellect-ai/prime-rl-perfrouter-rs-buffer-cap
MAIN_WORKTREE=$HOME/github/PrimeIntellect-ai/prime-rl-main
RUNS_DIR=$HOME/tmp/router-rs-cap/runs
TRACES_DIR=$HOME/tmp/router-rs-cap/traces

declare -A EP=([fsdp16]=1 [ep8]=8)
# variant -> "router_dtype fullgraph cap" (cap "-" means unset, launched from main)
declare -A VARIANT=(
  [fp32-cap1]="float32 false 1"
  [fp32-cap2]="float32 false 2"
  [fp32-cap2-r2]="float32 false 2"
  [fp32-cap3]="float32 false 3"
  [fp32-main]="float32 false -"
  [bf16fg-cap1]="bfloat16 true 1"
  [bf16fg-cap2]="bfloat16 true 2"
  [bf16-cap1]="bfloat16 false 1"
  [bf16-cap2]="bfloat16 false 2"
)

# scenario -> "seq_len ac max_steps num_layers"; s8k-noac at 24 layers peaks at 137 GiB and thrashes the allocator
declare -A SCENARIOS=(
  [s4k-ac]="4096 full 40 24"
  [s4k-noac]="4096 off 40 24"
  [s8k-noac]="8192 off 40 16"
  [s1k-noac]="1024 off 40 24"
)

layout=$1
variant=$2
shift 2
ep=${EP[$layout]}
read -r router fullgraph cap <<<"${VARIANT[$variant]}"

name="$layout-$variant"
group=$layout
scenario_args=()
scenario_tags=""
if [ -n "${SCENARIO:-}" ]; then
  read -r seq_len ac max_steps num_layers <<<"${SCENARIOS[$SCENARIO]}"
  name="$SCENARIO-$name"
  group="$SCENARIO-$layout"
  scenario_args=(--data.seq-len "$seq_len" --max-steps "$max_steps" --model.debug.num-layers "$num_layers")
  if [ "$ac" = "off" ]; then
    scenario_args+=(--model.ac None)
  fi
  scenario_tags=",\"$SCENARIO\",\"seq$seq_len\",\"ac-$ac\",\"layers$num_layers\""
fi
worktree=$BRANCH_WORKTREE
cap_args=(--model.reduce-scatter-max-input-buffers "$cap")
if [ "$cap" = "-" ]; then
  worktree=$MAIN_WORKTREE
  cap_args=()
fi

trace_args=()
tags="\"$layout\",\"router-$router\",\"fullgraph-$fullgraph\",\"cap-$cap\"$scenario_tags"
if [ "${1:-}" = "--trace" ]; then
  shift
  name="$name-trace"
  trace_args=(--trace-path "$TRACES_DIR/$name" --max-steps 6)
  tags="$tags,\"traced\""
fi

compile_args=()
if [ "$fullgraph" = "true" ]; then
  compile_args=(--model.compile.fullgraph)
fi

cd "$worktree"
env -u HF_HOME PRL_OUTPUT_DIR="$RUNS_DIR" uv run --no-sync sft @ "$AB_DIR/sft-base.toml" \
  --run.name "$name" \
  --model.ep "$ep" --model.moe-router-dtype "$router" \
  "${compile_args[@]}" "${cap_args[@]}" "${scenario_args[@]}" "${trace_args[@]}" \
  --monitors.wandb.name "$name" --monitors.wandb.group "$group" \
  --monitors.wandb.tags "[$tags]" \
  --no-dashboard "$@"
