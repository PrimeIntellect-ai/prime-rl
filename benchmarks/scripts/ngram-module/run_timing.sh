#!/usr/bin/env bash
# Interleave main and branch runs of the full-AC 8-GPU cells on one node to compare step time: run_timing.sh <main worktree> <main venv> <branch worktree> <branch venv> <root>
set -uo pipefail

main_worktree=$1
main_venv=$2
branch_worktree=$3
branch_venv=$4
root=$5
script_dir=$(cd "$(dirname "$0")" && pwd)

for repeat in 0 1 2; do
    for cell in 5 7; do
        uv run --script "$script_dir/run_ab.py" main "$main_worktree" "$cell" "$repeat" --venv "$main_venv" --root "$root"
        uv run --script "$script_dir/run_ab.py" final "$branch_worktree" "$cell" "$repeat" --venv "$branch_venv" --root "$root"
    done
done
