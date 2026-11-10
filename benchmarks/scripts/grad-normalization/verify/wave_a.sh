#!/usr/bin/env bash
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="${GN_WORKTREES:?set GN_WORKTREES to the worktree prefix; <prefix>-main and <prefix>-fix must exist}"
ROOT="${GN_VERIFY_OUT:-$PWD/grad-norm-verify}"
OUT="$ROOT/rl"
SFT="$ROOT/sft"
LOGS="$ROOT/logs"
mkdir -p "$OUT" "$SFT" "$LOGS"

arm() {
    local wt=$1 gpus=$2 layout=$3 name=$4
    bash "$H/launch.sh" --worktree "$W-$wt" --gpus "$gpus" --layout "$layout" --out "$OUT" --name "$name" \
        --grad-dump-step 1 --fake > "$LOGS/$name.log" 2>&1
    echo "$name exit $?"
}

arm main 0 single main-single &
arm main 1 single main-single-rep &
arm fix 2 single fix-single &
arm main 3,4 cp2 main-cp2 &
arm fix 5,6 cp2 fix-cp2 &
(cd "$W-fix" && CUDA_VISIBLE_DEVICES=7 uv run --no-sync pytest \
    "tests/unit/train/rl/test_loss.py::test_ce_component_matches_masked_nll" \
    "tests/unit/train/rl/test_loss.py::test_overlapping_components_sum" -q > "$LOGS/unit.log" 2>&1; echo "unit exit $?") &
wait
