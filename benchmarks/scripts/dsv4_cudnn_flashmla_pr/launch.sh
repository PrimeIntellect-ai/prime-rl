#!/usr/bin/env bash
# Run DSv4 dsa_backend A/B arms serially on one exclusive node held by SLURM job JOBID, each from a frozen
# detached worktree at its SHA with fresh per-run compile caches on the node's local /tmp, since ranks sharing a
# Triton cache on NFS race on its files.
#
# usage (login node; every heavy step runs on the job's node through srun):
#   launch.sh JOBID ARM [ARM ...] [-- EXTRA_SFT_ARGS ...]
#
# ARM is LABEL=SHA:BACKEND, e.g. A=7c9d4744c:tilelang. The run is named after LABEL, SHA and BACKEND.
#
# Example: a 3-step smoke check of the candidate, then the agreed matrix:
#   launch.sh 3935 smoke=50ca479a1:cudnn_flashmla -- --max-steps 3
#   launch.sh 3935 A=7c9d4744c:tilelang B=<tip>:tilelang B2=<tip>:tilelang C=<tip>:cudnn_flashmla ...
set -euo pipefail

JOBID=$1
shift
ARMS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  ARMS+=("$1")
  shift
done
[ "${1:-}" = "--" ] && shift
EXTRA_ARGS=("$@")

HERE=$(cd "$(dirname "$0")" && pwd)
SEED_WORKTREE=$(git -C "$HERE" rev-parse --show-toplevel)
REPO_PARENT=$HOME/github/PrimeIntellect-ai
OUTPUT_ROOT=/home/garrett/prl_output_dir/dsv4-cfm-pr
CACHE_ROOT=/tmp/dsv4-cfm-pr-$USER
CONFIG_ROOT=$HOME/tmp/dsv4-cfm-pr/configs
CONFIG_SOURCE=$HERE/sft.toml

on_node() {
  srun --jobid="$JOBID" --overlap -N1 -n1 bash -c "$1"
}

# Offline: clone each submodule from this worktree's checkout, at the commit the frozen SHA records.
seed_submodules() {
  local frozen=$1
  for path in $(git -C "$frozen" config --file .gitmodules --get-regexp path | awk '{print $2}'); do
    local pinned
    pinned=$(git -C "$frozen" ls-tree HEAD "$path" | awk '{print $3}')
    rm -rf "${frozen:?}/$path"
    git clone -q --no-checkout "$SEED_WORKTREE/$path" "$frozen/$path"
    git -C "$frozen/$path" checkout -q "$pinned"
  done
}

frozen_worktree() {
  local sha=$1 frozen=$REPO_PARENT/prime-rl-run-dsv4-cfm-$1
  if [ ! -d "$frozen" ]; then
    git -C "$SEED_WORKTREE" worktree add -q --detach "$frozen" "$sha"
    seed_submodules "$frozen"
    on_node "cd $frozen && uv sync --all-extras --all-packages" >&2
  fi
  echo "$frozen"
}

for arm in "${ARMS[@]}"; do
  label=${arm%%=*}
  spec=${arm#*=}
  sha=$(git -C "$SEED_WORKTREE" rev-parse --short=9 "${spec%%:*}")
  backend=${spec#*:}
  frozen=$(frozen_worktree "$sha")
  name="dsv4-flash-6L-sft-65k-cp8-ep8-$backend-$label-$sha"
  tags="[\"dsa-backend-ab\",\"compile-once\",\"$backend\",\"arm-$label\",\"6-layers\",\"seq65536\",\"cp8\",\"ep8\",\"intellect-4-sft-swe\",\"$sha\"]"
  cache_dir=$CACHE_ROOT/$name
  config=$CONFIG_ROOT/$name.toml
  mkdir -p "$CONFIG_ROOT"
  cp "$CONFIG_SOURCE" "$config"
  env_vars="{\"TILELANG_CACHE_DIR\":\"$cache_dir/tilelang\",\"TRITON_CACHE_DIR\":\"$cache_dir/triton\","
  env_vars+="\"TORCHINDUCTOR_CACHE_DIR\":\"$cache_dir/inductor\",\"CUDA_CACHE_PATH\":\"$cache_dir/cuda\","
  env_vars+="\"TORCH_LOGS\":\"recompiles\"}"
  echo "[$(date -u +%FT%TZ)] $name from $frozen" >&2
  on_node "rm -rf $cache_dir && mkdir -p $cache_dir/{tilelang,triton,inductor,cuda} && cd $frozen && env -u HF_HOME \
    HF_HUB_CACHE=/home/huggingface/hub \
    PRL_OUTPUT_DIR=$OUTPUT_ROOT \
    uv run --no-sync sft @ $config \
      --run.name $name \
      --model.dsa-backend $backend \
      --env-vars '$env_vars' \
      --monitors.wandb.name $name \
      --monitors.wandb.group dsv4-flash-6L-dsa-backend-compile-once \
      --monitors.wandb.tags '$tags' \
      --no-dashboard ${EXTRA_ARGS[*]}" || echo "[$(date -u +%FT%TZ)] $name FAILED" >&2
done
