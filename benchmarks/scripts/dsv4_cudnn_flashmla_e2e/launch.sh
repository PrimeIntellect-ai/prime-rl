#!/usr/bin/env bash
# Run DSv4 dsa_backend A/B arms serially on one exclusive node held by SLURM job JOBID, each from a frozen
# detached worktree at SHA with fresh per-run compile caches.
#
# usage (login node; every heavy step runs on the job's node through srun):
#   launch.sh JOBID SHA ARM [ARM ...] [-- EXTRA_SFT_ARGS ...]
#
# ARM is one of:
#   tilelang, cudnn_flashmla        timing arms, max_steps from sft.toml (20)
#   tilelang-r2, cudnn_flashmla-r2  same-code repeats, the noise floor
#   trace-tilelang, trace-cudnn_flashmla  5-step torch.profiler runs (--trace-path)
#
# Example: the agreed matrix, then a smoke check of one arm with 3 steps:
#   launch.sh 3572 6c67cb057 tilelang cudnn_flashmla tilelang-r2 trace-tilelang trace-cudnn_flashmla
#   launch.sh 3572 6c67cb057 cudnn_flashmla -- --max-steps 3
set -euo pipefail

JOBID=$1
SHA=$(git -C "$(dirname "$0")" rev-parse --short=9 "$2")
shift 2
ARMS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  ARMS+=("$1")
  shift
done
[ "${1:-}" = "--" ] && shift
EXTRA_ARGS=("$@")

REPO_PARENT=$HOME/github/PrimeIntellect-ai
FROZEN=$REPO_PARENT/prime-rl-run-dsv4-cfm-$SHA
OUTPUT_ROOT=/home/garrett/prl_output_dir/dsv4-cfm-e2e
CACHE_ROOT=$HOME/tmp/profiling/caches/dsv4-cfm-e2e
TRACE_ROOT=$HOME/tmp/profiling/dsv4-cfm-e2e/traces
CONFIG=benchmarks/scripts/dsv4_cudnn_flashmla_e2e/sft.toml

on_node() {
  srun --jobid="$JOBID" --overlap -N1 -n1 bash -c "$1"
}

if [ ! -d "$FROZEN" ]; then
  git -C "$(dirname "$0")" worktree add --detach "$FROZEN" "$SHA"
  git -C "$FROZEN" submodule update --init --recursive
  on_node "cd $FROZEN && uv sync --all-extras --all-packages"
fi

for arm in "${ARMS[@]}"; do
  backend=${arm#trace-}
  backend=${backend%-r2}
  run_args=()
  if [[ $arm == trace-* ]]; then
    run_args=(--max-steps 5 --trace-path "$TRACE_ROOT/$arm-$SHA")
  fi
  name="dsv4-flash-6L-sft-65k-cp8-ep8-$arm-$SHA"
  tags="[\"dsa-backend-ab\",\"$backend\",\"$arm\",\"6-layers\",\"seq65536\",\"cp8\",\"ep8\",\"intellect-4-sft-swe\",\"$SHA\"]"
  cache_dir=$CACHE_ROOT/$name
  rm -rf "$cache_dir"
  mkdir -p "$cache_dir"/{tilelang,triton,inductor,cute_dsl,cuda}
  echo "[$(date -u +%FT%TZ)] $name from $FROZEN" >&2
  on_node "cd $FROZEN && env -u HF_HOME \
    HF_HUB_CACHE=/home/huggingface/hub \
    PRL_OUTPUT_DIR=$OUTPUT_ROOT \
    TILELANG_CACHE_DIR=$cache_dir/tilelang \
    TRITON_CACHE_DIR=$cache_dir/triton \
    TORCHINDUCTOR_CACHE_DIR=$cache_dir/inductor \
    CUTE_DSL_CACHE_DIR=$cache_dir/cute_dsl \
    CUDA_CACHE_PATH=$cache_dir/cuda \
    uv run --no-sync sft @ $CONFIG \
      --run.name $name \
      --model.dsa-backend $backend \
      --monitors.wandb.name $name \
      --monitors.wandb.group dsv4-flash-6L-dsa-backend-ab-$SHA \
      --monitors.wandb.tags '$tags' \
      --no-dashboard ${run_args[*]} ${EXTRA_ARGS[*]}" || echo "[$(date -u +%FT%TZ)] $name FAILED" >&2
done
