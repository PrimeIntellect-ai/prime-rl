#!/usr/bin/env bash
# Usage: make_arm.sh <arm-name> <sha>. Detached worktree at <sha> with diag.patch and data.patch applied, under ~/tmp.
set -euo pipefail
name=$1; sha=$2
harness=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$harness" rev-parse --show-toplevel)
arm=$HOME/tmp/sft_pad_to_cp/arms/$name
git -C "$repo" worktree add --detach "$arm" "$sha"
cd "$arm"
git submodule update --init --recursive
uv sync --all-extras --all-packages
git apply "$harness/diag.patch" "$harness/data.patch"
echo "arm $name at $(git rev-parse --short HEAD) + diag.patch + data.patch -> $arm"
