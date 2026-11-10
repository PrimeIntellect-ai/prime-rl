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
sft() {
    local wt=$1 gpus=$2 n=$3 name=$4
    (cd "$W-$wt" && CUDA_VISIBLE_DEVICES=$gpus uv run --no-sync sft @ "$H/verify/sft_bs6.toml" \
        --output-dir "$SFT" --run.name "$name" --deployment.num-train-gpus "$n" --no-monitors.prime \
        > "$LOGS/$name.log" 2>&1)
    echo "$name exit $?"
}
sft main 0 1 sft6-main-1gpu &
sft fix 1 1 sft6-fix-1gpu &
sft main 2,3 2 sft6-main-dp2 &
sft fix 4,5 2 sft6-fix-dp2 &
arm main 6 single mc-main-single-rep &
arm fix 7 single mc-fix-single-rep &
wait
