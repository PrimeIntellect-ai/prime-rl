#!/usr/bin/env bash

set -euo pipefail

service=${1:-opd-gap-base-band40-hosted-r03.service}
results=${2:-/home/ubuntu/prime-rl/evals/genagent-base-band40-k32-hosted-r01-20260715/results.jsonl}
state_file=${3:-/home/ubuntu/opd-gap-base-band40-hosted-state.json}
expected=1280
last_count=-1
last_progress=$(date +%s)

write_state() {
  local phase=$1
  local rows=$2
  local errors=$3
  local detail=${4:-}
  jq -n \
    --arg phase "$phase" \
    --arg detail "$detail" \
    --arg service "$service" \
    --arg results "$results" \
    --argjson rows "$rows" \
    --argjson errors "$errors" \
    --argjson expected "$expected" \
    '{phase:$phase,detail:$detail,service:$service,results:$results,rows:$rows,expected_rows:$expected,error_rows:$errors,updated_at_utc:(now|todateiso8601)}' \
    >"$state_file.tmp"
  mv "$state_file.tmp" "$state_file"
}

while true; do
  rows=0
  errors=0
  if [[ -s "$results" ]]; then
    read -r rows errors < <(
      jq -s '[length, (map(select((.errors | length) > 0)) | length)] | @tsv' "$results" -r
    )
  fi
  now=$(date +%s)
  if (( rows != last_count )); then
    last_count=$rows
    last_progress=$now
  fi

  if (( errors > 0 )); then
    write_state failed "$rows" "$errors" "completed rows contain errors after retries"
    systemctl --user stop "$service" || true
    exit 1
  fi
  if (( rows == expected )); then
    write_state complete "$rows" "$errors" "exact row and error gates passed"
    exit 0
  fi
  if (( rows > expected )); then
    write_state failed "$rows" "$errors" "row count exceeds expected total"
    systemctl --user stop "$service" || true
    exit 1
  fi

  active=$(systemctl --user is-active "$service" 2>/dev/null || true)
  if [[ "$active" != "active" ]]; then
    write_state failed "$rows" "$errors" "service exited before exact completion"
    exit 1
  fi
  if (( now - last_progress > 1200 )); then
    write_state failed "$rows" "$errors" "no new result row for 20 minutes"
    systemctl --user stop "$service" || true
    exit 1
  fi
  write_state running "$rows" "$errors" "service active"
  sleep 30
done
