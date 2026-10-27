#!/usr/bin/env bash
# Run the full-AC parity cells serially:
# run_parity.sh <main-worktree> <main-venv> <branch-worktree> <branch-venv> <root> <checkpoint>
set -uo pipefail

main_worktree=$1
main_venv=$2
branch_worktree=$3
branch_venv=$4
root=$5
checkpoint=$(realpath "$6")
name=$(basename "$checkpoint")
script_dir=$(cd "$(dirname "$0")" && pwd)

run() {
    local arm=$1 worktree=$2 venv=$3
    shift 3
    uv run --script "$script_dir/run_ab.py" "$arm" "$worktree" --venv "$venv" \
        --root "$root/runs-$name" --conversion-root "$root/conv-$name" --extra="--model.name $checkpoint" "$@"
}

run main "$main_worktree" "$main_venv" 5 0
run branch "$branch_worktree" "$branch_venv" 5 0
run branch "$branch_worktree" "$branch_venv" 6 0
run main "$main_worktree" "$main_venv" 6 0
