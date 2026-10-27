#!/usr/bin/env bash
# Run one node's share of the A/B matrix for one arm, serially: run_matrix.sh <arm> <worktree> <venv> <a|b>
set -uo pipefail

arm=$1
worktree=$2
venv=$3
node_group=$4
script_dir=$(cd "$(dirname "$0")" && pwd)

run() {
    uv run --script "$script_dir/run_ab.py" "$arm" "$worktree" --venv "$venv" "$@"
}

case $node_group in
    a)
        for cell in 1 2 3 4 7; do
            run "$cell" 0
            run "$cell" 1
        done
        run 3 0 --graph-breaks --steps 8
        ;;
    b)
        run 5 0
        run 5 1
        run 6 0
        if [ "$arm" != main ]; then
            run 6 1
        fi
        run 8 0
        run 8 1
        run 8 0 --extra='--model.ac None' --tag noac
        run 8 1 --extra='--model.ac None' --tag noac
        run 5 0 --trace
        run 8 0 --extra='--model.ac None' --tag noac --graph-breaks --steps 8
        ;;
esac
