#!/usr/bin/env bash
# Confirmation, activation-offload and stale-cache runs for one arm, serially: run_followups.sh <arm> <worktree> <venv>
set -uo pipefail

arm=$1
worktree=$2
venv=$3
script_dir=$(cd "$(dirname "$0")" && pwd)

run() {
    uv run --script "$script_dir/run_ab.py" "$arm" "$worktree" --venv "$venv" "$@"
}

run 5 0
run 6 0
run 8 0 --extra='--model.ac None' --tag noac
run 1 0
run 5 0 --extra='--model.ac-offloading' --tag acoffload
run 5 0 --extra='--model.conversion-dir /home/garrett/tmp/ngram-module/conv/main' --tag stale --steps 2
