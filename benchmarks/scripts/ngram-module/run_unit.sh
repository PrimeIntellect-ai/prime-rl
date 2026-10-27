#!/usr/bin/env bash
# Run one arm's unit tests from its frozen worktree: run_unit.sh <worktree> <venv> <out-dir> <cpu|gpu> [pytest args]
set -uo pipefail

worktree=$(realpath "$1")
venv=$(realpath "$2")
out_dir=$(realpath -m "$3")
suite=$4
shift 4

mkdir -p "$out_dir" "/tmp/$USER"
cd "$worktree"
export VIRTUAL_ENV=$venv
PYTEST_OUTPUT_DIR=$(mktemp -d "/tmp/$USER/ngram-unit-XXXXXX")
export PYTEST_OUTPUT_DIR

run_pytest() {
    local name=$1
    shift
    uv run --active --no-sync pytest -p no:cacheprovider -rfEs --junitxml="$out_dir/$name.xml" "$@" > "$out_dir/$name.log" 2>&1
    echo "$name exit $?"
}

case $suite in
    cpu)
        run_pytest cpu tests/unit -m "not gpu" "$@"
        ;;
    gpu)
        uv run --active --no-sync torchrun --nproc-per-node=4 --log-dir "$out_dir/ngram-lookup-ranks" --redirects 3 --tee 3 \
            -m pytest -p no:cacheprovider -rfEs tests/unit/train/models/test_ngram_lookup.py > "$out_dir/ngram-lookup.log" 2>&1
        echo "ngram-lookup exit $?"
        run_pytest qwen3_8_flash_next tests/unit/train/models/qwen3_8_flash_next "$@"
        run_pytest gpu-models -m gpu tests/unit/train/models tests/unit/train/test_model.py \
            --ignore=tests/unit/train/models/qwen3_8_flash_next "$@"
        ;;
esac
