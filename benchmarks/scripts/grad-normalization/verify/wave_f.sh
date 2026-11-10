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
    local mode=$1 wt=$2 gpus=$3 layout=$4 name=$5
    GN_MC_MODE=$mode bash "$H/launch.sh" --worktree "$W-$wt" --gpus "$gpus" --layout "$layout" --out "$OUT" --name "$name" \
        --grad-dump-step 1 --fake > "$LOGS/$name.log" 2>&1
    echo "$name exit $?"
}


arm rl1000 main 0 single rl1000-main-single &
arm rl1000 fix 1 single rl1000-fix-single &
arm ce1000 main 2 single ce1000-main-single &
arm ce1000 fix 3 single ce1000-fix-single &
arm mod main 4 single mc-main-single-rep2 &


wait
