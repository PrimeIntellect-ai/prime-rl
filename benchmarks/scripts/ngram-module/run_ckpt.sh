#!/usr/bin/env bash
# Branch-only checkpoint round trip on cell 5: save at step 10 of 20, resume a second run from it, export step 10 to HF.
# run_ckpt.sh <worktree> <venv> <root> <checkpoint>
set -uo pipefail

worktree=$(realpath "$1")
venv=$(realpath "$2")
root=$3
checkpoint=$(realpath "$4")
name=$(basename "$checkpoint")
script_dir=$(cd "$(dirname "$0")" && pwd)
step_dir=$root/runs-$name/branch/cell5-ckpt/r0/ab/checkpoints/step_10

run() {
    uv run --script "$script_dir/run_ab.py" branch "$worktree" --venv "$venv" \
        --root "$root/runs-$name" --conversion-root "$root/conv-$name" --extra="--model.name $checkpoint" 5 0 "$@"
}

run --tag ckpt --extra="--ckpt.interval 10"
run --tag resume --extra="--resume.dir $step_dir"

cd "$worktree"
export VIRTUAL_ENV=$venv
uv run --active --no-sync torchrun --nproc-per-node=8 tools/convert_dcp_to_bf16.py "$step_dir" \
    > "$root/logs/ckpt-export.log" 2>&1
echo "export exit $?"
uv run --active --no-sync python "$script_dir/check_ngram_ckpt_names.py" "$step_dir" "$step_dir/weights" "$checkpoint" \
    > "$root/logs/ckpt-names.log" 2>&1
echo "names exit $?"
