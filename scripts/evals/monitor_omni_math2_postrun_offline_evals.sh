#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <status-path> <comparison-path> <job-id> [<job-id> ...]" >&2
  exit 2
fi

status_path=$1
comparison_path=$2
shift 2
job_ids=("$@")

repo_root=$(git rev-parse --show-toplevel)
mkdir -p "$(dirname "$status_path")"

write_status() {
  {
    echo "# OmniMath2 Postrun Eval Monitor"
    echo
    echo "- updated_utc: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`"
    echo "- jobs: \`${job_ids[*]}\`"
    echo "- comparison: \`$comparison_path\`"
    echo
    echo "## Queue"
    echo
    squeue -j "$(IFS=,; echo "${job_ids[*]}")" -o "%.18i %.40j %.8T %.10M %.9l %.6D %R" || true
    echo
    echo "## Accounting"
    echo
    sacct -j "$(IFS=,; echo "${job_ids[*]}")" \
      --format=JobID,JobName%36,State,Elapsed,ExitCode,NodeList%80 -P || true
    echo
    echo "## Current Summaries"
    echo
    find "$repo_root/outputs/omni_math2_rlvr_canary" \
      -path '*offline_eval_600x8_7node_clean*' \
      -name summary.json \
      -print | sort || true
  } > "$status_path.tmp"
  mv "$status_path.tmp" "$status_path"
}

last_summary_count=-1

while true; do
  write_status

  summary_count=$(
    find "$repo_root/outputs/omni_math2_rlvr_canary" \
      -path '*offline_eval_600x8_7node_clean*' \
      -name summary.json \
      -print | wc -l
  )
  if [[ "$summary_count" != "$last_summary_count" ]]; then
    uv run --no-sync python scripts/evals/compare_omni_math2_offline_evals.py \
      --output "$comparison_path" || true
    last_summary_count=$summary_count
    write_status
  fi

  active_jobs=$(squeue -h -j "$(IFS=,; echo "${job_ids[*]}")" | wc -l)
  if [[ "$active_jobs" -eq 0 ]]; then
    uv run --no-sync python scripts/evals/compare_omni_math2_offline_evals.py \
      --output "$comparison_path" || true
    write_status
    break
  fi

  sleep 120
done
