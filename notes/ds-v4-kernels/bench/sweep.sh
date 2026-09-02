#!/usr/bin/env bash
# Drive `profile_ds_v4.py` over the (module, seq_len, mode) grid, eight modules at a time.
#
# One subprocess per measured point, because the sweep is designed to hit OOM and a process that
# has raised `OutOfMemoryError` reports garbage for every peak after it. One GPU handles one
# module's whole series serially: co-locating two memory-profiling processes on a device makes
# both peaks meaningless.
#
# Idempotent. A point whose JSON already exists is skipped, so an interrupted sweep resumes by
# being re-run.
#
#   ./notes/ds-v4-kernels/bench/sweep.sh
#
set -uo pipefail
cd "$(dirname "$0")/../../.."

RAW=notes/ds-v4-kernels/bench/raw
LOGS=outputs/ds-v4-kernels/logs
HARNESS=notes/ds-v4-kernels/bench/profile_ds_v4.py
LENS=(2048 4096 8192 12288 16384 24576 32768)
MODULES=(attn-csa attn-hca attn-sliding indexer-scorer indexer compressor-csa hyperconnection compressor-hca rmsnorm rotary packed-context)

mkdir -p "$RAW" "$LOGS"

status_of() {
  # No jq on this box; the harness writes indented JSON, so one grep is enough.
  grep -o '"status": "[a-z]*"' "$1" | head -1 | cut -d'"' -f4
}

run_point() {
  local gpu=$1 module=$2 t=$3 mode=$4
  local tag="${module}__t${t}__${mode}"
  local out="$RAW/${tag}.json"
  if [[ -f "$out" ]]; then
    status_of "$out"
    return
  fi
  # `do_bench`'s warmup/rep are milliseconds of total work, not iteration counts. The default
  # rep=100 buys one or two reps at these shapes, which collapses the quantiles onto the mean and
  # lets a single scheduling hiccup move the median by 20%. 3000 ms of reps holds it to ~0.1%.
  local budget=()
  [[ "$mode" == "timing" ]] && budget=(--warmup 300 --rep 3000)
  CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync "$HARNESS" "$module" "$t" \
    --mode "$mode" "${budget[@]}" --out "$out" > "$LOGS/${tag}.log" 2>&1
  if [[ ! -f "$out" ]]; then
    echo "crash"
    return
  fi
  status_of "$out"
}

run_module() {
  local gpu=$1 module=$2
  local survived=()
  for t in "${LENS[@]}"; do
    local status
    status=$(run_point "$gpu" "$module" "$t" memory)
    echo "[gpu$gpu] $module t=$t memory -> $status"
    if [[ "$status" == "ok" ]]; then
      survived+=("$t")
    else
      # First OOM in the series is this module's ceiling; escalating further measures nothing.
      break
    fi
  done
  for t in "${survived[@]}"; do
    local status
    status=$(run_point "$gpu" "$module" "$t" timing)
    echo "[gpu$gpu] $module t=$t timing -> $status"
  done
}

gpu=0
for module in "${MODULES[@]}"; do
  run_module "$gpu" "$module" &
  gpu=$(((gpu + 1) % 8))
  # More modules than GPUs: let the first wave finish before starting the wrap-around, so no two
  # processes ever share a device.
  if ((gpu == 0)); then wait; fi
done
wait
echo "sweep complete: $(ls "$RAW" | wc -l) points in $RAW"
