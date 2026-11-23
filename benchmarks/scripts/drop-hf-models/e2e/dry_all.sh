#!/usr/bin/env bash
set -uo pipefail

HARNESS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HARNESS/env.sh"
source "$HARNESS/lib.sh"

[ $# -eq 1 ] || { echo "usage: dry_all.sh <arm>" >&2; exit 2; }
arm=$1
mkdir -p "$E2E_OUT/dry/$arm"
pids=()
cells=()
for cell in "${ALL_CELLS[@]}"; do
  if [[ $cell == *-resume ]] && [ ! -d "$E2E_OUT/main1/${cell%-resume}-write/checkpoints/step_10" ]; then
    continue
  fi
  bash "$HARNESS/run_cell.sh" "$arm" "$cell" --dry-run >"$E2E_OUT/dry/$arm/$cell.log" 2>&1 &
  pids+=($!)
  cells+=("$cell")
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "dry-run ok   ${cells[$i]}"
  else
    echo "dry-run FAIL ${cells[$i]} (see $E2E_OUT/dry/$arm/${cells[$i]}.log)"
    failed=1
  fi
done
for name in rl-dense-nccl rl-moe-nccl; do
  trainer=$E2E_OUT/dry/$arm/$name/configs/latest/resolved/trainer.json
  [ -f "$trainer" ] && echo "$name inference_world_size: $(grep -o '"inference_world_size": *[0-9]*' "$trainer")"
done
exit "$failed"
