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
    local mode=$1 rescale=$2 wt=$3 gpus=$4 layout=$5 name=$6
    GN_MC_MODE=$mode GN_LOSS_RESCALE=$rescale bash "$H/launch.sh" --worktree "$W-$wt" --gpus "$gpus" --layout "$layout" \
        --out "$OUT" --name "$name" --grad-dump-step 1 --fake > "$LOGS/$name.log" 2>&1
    echo "$name exit $?"
}
arm mod 3 main 0 single mc-main-x3-single &
arm mod 1.7 main 1 single mc-main-x1.7-single &
arm mod 3 fix 2 single mc-fix-x3-single &
arm ce1000 3 main 3 single ce1000-main-x3-single &
arm none 3 main 4 single main-x3-single &
arm none 4 main 5 single main-x4-single &
arm pow2 3 main 6 single pow2-main-x3-single &
arm mod 4 main 7 single mc-main-x4-single &
wait
