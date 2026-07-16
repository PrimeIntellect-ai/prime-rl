#!/usr/bin/env bash

set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "failure" && "$mode" != "success" ]]; then
    echo "usage: $0 {failure|success}" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
output_dir="$repo_root/outputs/repro-main-$mode"
repro_dir="$output_dir/race_repro"

# These are fixed, mode-specific paths under the current worktree. Cleaning
# them cannot affect another run unless that run deliberately uses this script.
rm -rf -- "$output_dir"
mkdir -p "$output_dir"

export PYTHONPATH="$repo_root/src:$repo_root/packages/prime-rl-configs/src${PYTHONPATH:+:$PYTHONPATH}"
export PRIME_RL_FINAL_RACE_REPRO="$mode"
export PRIME_RL_FINAL_RACE_REPRO_DIR="$repro_dir"

uv_bin="${UV_BIN:-uv}"
uv_args=(run --frozen)
if [[ -n "${PRIME_RL_REPRO_UV_PROJECT:-}" ]]; then
    # Useful for a git worktree sharing an already-synced environment.
    uv_args+=(--project "$PRIME_RL_REPRO_UV_PROJECT" --no-sync)
fi

echo "mode=$mode"
echo "output_dir=$output_dir"
echo "repro_dir=$repro_dir"

"$uv_bin" "${uv_args[@]}" rl \
    @ "$repo_root/examples/reverse_text/rl.toml" \
    --output-dir "$output_dir" \
    --no-wandb \
    2>&1 | tee "$output_dir/launcher.log"
