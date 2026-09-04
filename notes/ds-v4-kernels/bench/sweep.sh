#!/usr/bin/env bash
# Drive `profile_ds_v4.py` over the (module, attn impl, seq_len, mode) grid, one job per visible GPU.
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
# Every axis is overridable from the environment, so one module or one implementation can be
# re-swept without editing this file:
#
#   MODULES=attn-csa IMPLS="eager gather kernel" LENS="2048 4096" ./notes/ds-v4-kernels/bench/sweep.sh
#
set -uo pipefail
cd "$(dirname "$0")/../../.."

RAW=notes/ds-v4-kernels/bench/raw
LOGS=outputs/ds-v4-kernels/logs
HARNESS=notes/ds-v4-kernels/bench/profile_ds_v4.py
read -r -a LENS <<< "${LENS:-2048 4096 8192 12288 16384 24576 32768}"
read -r -a MODULES <<< "${MODULES:-attn-csa attn-hca attn-sliding indexer-scorer indexer compressor-csa hyperconnection compressor-hca rmsnorm rotary packed-context}"
# Only the CSA modules have more than one implementation, so the default is the single one the
# harness itself defaults to; widen it explicitly to compare paths.
read -r -a IMPLS <<< "${IMPLS:-kernel}"
# A caller who restricts the pool means it: two memory-profiling processes on one device would
# report garbage peaks for both. Take the inherited ids verbatim, in their order, and only ask the
# driver for the full set when nothing was inherited.
GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  read -r -a GPUS <<< "$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
  ((${#GPUS[@]})) || GPUS=(0 1 2 3 4 5 6 7)
fi

mkdir -p "$RAW" "$LOGS"

status_of() {
  # No jq on this box; the harness writes indented JSON, so one grep is enough.
  grep -o '"status": "[a-z]*"' "$1" | head -1 | cut -d'"' -f4
}

run_point() {
  local gpu=$1 module=$2 t=$3 mode=$4 impl=$5
  # `eager` keeps the bare tag the pre-impl sweep wrote, since those files are all eager: that is
  # what lets an existing point still resume. The other implementations take a segment of their
  # own, without which the second impl would skip every point the first one already wrote.
  local tag="${module}__t${t}__${mode}"
  [[ "$impl" != "eager" ]] && tag="${module}__t${t}__${impl}__${mode}"
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
    --mode "$mode" --attn-impl "$impl" "${budget[@]}" --out "$out" > "$LOGS/${tag}.log" 2>&1
  if [[ ! -f "$out" ]]; then
    echo "crash"
    return
  fi
  status_of "$out"
}

run_module() {
  local gpu=$1 module=$2 impl=$3
  local survived=()
  for t in "${LENS[@]}"; do
    local status
    status=$(run_point "$gpu" "$module" "$t" memory "$impl")
    echo "[gpu$gpu] $module/$impl t=$t memory -> $status"
    if [[ "$status" == "ok" ]]; then
      survived+=("$t")
    else
      # First OOM in the series is this module's ceiling; escalating further measures nothing.
      break
    fi
  done
  for t in "${survived[@]}"; do
    local status
    status=$(run_point "$gpu" "$module" "$t" timing "$impl")
    echo "[gpu$gpu] $module/$impl t=$t timing -> $status"
  done
}

slot=0
for module in "${MODULES[@]}"; do
  for impl in "${IMPLS[@]}"; do
    run_module "${GPUS[slot]}" "$module" "$impl" &
    slot=$(((slot + 1) % ${#GPUS[@]}))
    # More jobs than GPUs: let the first wave finish before starting the wrap-around, so no two
    # processes ever share a device.
    if ((slot == 0)); then wait; fi
  done
done
wait
echo "sweep complete: $(ls "$RAW" | wc -l) points in $RAW"
