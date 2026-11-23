#!/usr/bin/env bash
set -euo pipefail

HARNESS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HARNESS/env.sh"
source "$HARNESS/lib.sh"

[ $# -eq 1 ] || { echo "usage: prep_node.sh <main1|pr1|pr2>" >&2; exit 2; }
worktree=$(arm_worktree "$1")
setup_uv "$worktree"
mkdir -p "$E2E_OUT" "$E2E_SCRATCH"

cd "$worktree"
if [ "$E2E_NODE_LOCAL_VENV" = 1 ]; then
  [ -d "$VIRTUAL_ENV" ] || uv venv "$VIRTUAL_ENV"
  uv sync --active --frozen --all-extras --all-packages
else
  uv sync --all-extras --all-packages
fi

if [ ! -f "$E2E_DATA/vlm-synth/train.jsonl" ]; then
  "$PY" "$HARNESS/make_vlm_dataset.py" "$E2E_DATA/vlm-synth"
fi
