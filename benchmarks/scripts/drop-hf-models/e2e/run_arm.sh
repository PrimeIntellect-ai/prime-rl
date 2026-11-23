#!/usr/bin/env bash
set -uo pipefail

HARNESS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HARNESS/env.sh"
source "$HARNESS/lib.sh"

[ $# -ge 2 ] || { echo "usage: run_arm.sh <main1|main2|pr1|pr2> <tier1|tier2|tier3|noise|cell>..." >&2; exit 2; }
arm=$1
shift
cells=()
for selector in "$@"; do
  case $selector in
    tier1) cells+=("${TIER1[@]}") ;;
    tier2) cells+=("${TIER2[@]}") ;;
    tier3) cells+=("${TIER3[@]}") ;;
    noise) cells+=("${NOISE[@]}") ;;
    *) cells+=("$selector") ;;
  esac
done

minutes_left() {
  local left days=0 hours=0 minutes=0
  left=$(squeue -j "$E2E_JOB" -h -o %L)
  [[ $left == *-* ]] && { days=${left%%-*}; left=${left#*-}; }
  IFS=: read -ra parts <<<"$left"
  case ${#parts[@]} in
    3) hours=${parts[0]}; minutes=${parts[1]} ;;
    2) minutes=${parts[0]} ;;
  esac
  echo $((10#$days * 1440 + 10#$hours * 60 + 10#$minutes))
}

mkdir -p "$E2E_OUT/$arm"
export_pids=()
for cell in "${cells[@]}"; do
  if [[ $cell == *-write ]] && [ "$arm" != main1 ]; then
    continue
  fi
  if [ -n "${E2E_JOB:-}" ] && [ "$(minutes_left)" -lt "${E2E_MIN_LEFT:-15}" ]; then
    echo "only $(minutes_left) min left on job $E2E_JOB; stopping before $cell"
    break
  fi
  bash "$HARNESS/run_cell.sh" "$arm" "$cell"
  if [[ $cell == ckpt-*-resume ]] && [ "$arm" != main2 ]; then
    ckpt=${cell%-resume}
    bash "$HARNESS/ckpt_export.sh" "$arm" "$ckpt" >"$E2E_OUT/$arm/$ckpt.export.log" 2>&1 &
    export_pids+=($!)
  fi
done
for pid in "${export_pids[@]}"; do
  wait "$pid"
done
cat "$E2E_OUT/$arm/status.tsv"
