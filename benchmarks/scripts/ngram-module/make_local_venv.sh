#!/usr/bin/env bash
# Build a node-local venv for one worktree, so trainer imports skip the slow /home filesystem.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <worktree> <venv-dir>" >&2
    exit 2
fi

worktree=$(realpath "$1")
venv=$(realpath -m "$2")

cd "$worktree"
uv venv --python "$(.venv/bin/python -c 'import platform; print(platform.python_version())')" "$venv"
VIRTUAL_ENV="$venv" uv sync --active --frozen --all-extras --all-packages
echo "$worktree" > "$venv/prime-rl-worktree"
