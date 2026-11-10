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
sft() {
    local wt=$1 gpus=$2 n=$3 name=$4
    (cd "$W-$wt" && CUDA_VISIBLE_DEVICES=$gpus uv run --no-sync sft @ "$H/verify/sft.toml" \
        --output-dir "$SFT" --run.name "$name" --deployment.num-train-gpus "$n" --no-monitors.prime \
        > "$LOGS/$name.log" 2>&1)
    echo "$name exit $?"
}

arm main 0,1 dp2 main-dp2 &
arm fix 2,3 dp2 fix-dp2 &
arm main 4,5 dp2 main-dp2-rep &
sft main 6 1 sft-main-1gpu &
sft fix 7 1 sft-fix-1gpu &
wait
