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
    GN_MC_MODE=mod bash "$H/launch.sh" --worktree "$W-$wt" --gpus "$gpus" --layout "$layout" --out "$OUT" --name "$name" \
        --grad-dump-step 1 --fake > "$LOGS/$name.log" 2>&1
    echo "$name exit $?"
}
arm main 0 single mc-main-single &
arm fix 1 single mc-fix-single &
arm main 2,3 dp2 mc-main-dp2 &
arm fix 4,5 dp2 mc-fix-dp2 &
wait
