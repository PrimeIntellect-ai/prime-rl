#!/usr/bin/env bash

set -euo pipefail

pod_id=${R40_POD_ID:-e87271e0b59843ce88a7181a5ce75c69}
prime=${R40_PRIME:-/home/ubuntu/.local/bin/prime}
key=${R40_SSH_KEY:-/home/ubuntu/.ssh/primeintellect_ed25519}
source_repo=${R40_SOURCE_REPO:-/home/ubuntu/prime-rl}
state=${R40_GEPA_STATE:-/home/ubuntu/opd-gap-r40-gepa-controller-state.json}
log=${R40_GEPA_LOG:-/home/ubuntu/opd-gap-r40-gepaplan-smoke.log}
config=configs/opd-gap/genagent-band000060-qwen35-opsd-gepaplan-pod-r40-smoke2.toml
output=outputs-genagent-opsd-1lp-d64-gepaplan-band000060-k8-tp4-pod-r40-smoke2-20260715

write_state() {
  jq -n --arg phase "$1" --arg detail "${2:-}" --arg pod_id "$pod_id" \
    '{phase:$phase,detail:$detail,pod_id:$pod_id,updated_at_utc:(now|todateiso8601)}' \
    >"$state.tmp"
  mv "$state.tmp" "$state"
}

fail() {
  code=$?
  write_state failed "controller exit $code"
  exit "$code"
}
trap fail ERR

status=$(env -u PRIME_API_KEY -u PRIME_TEAM_ID "$prime" pods status "$pod_id" --plain)
ssh_line=$(sed -n 's/^[[:space:]]*SSH[[:space:]]*//p' <<<"$status" | xargs)
[[ -n "$ssh_line" && "$ssh_line" != "N/A" ]]
remote=${ssh_line%% *}
port=$(awk '{for (i=1;i<=NF;i++) if ($i=="-p") print $(i+1)}' <<<"$ssh_line")
port=${port:-22}
ssh_args=(-i "$key" -p "$port" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
ssh_cmd="ssh -i $key -p $port -o StrictHostKeyChecking=no"
repo=/home/ubuntu/prime-rl

write_state syncing "syncing selected candidate 2 and template_path support"
for path in packages/prime-rl-configs src/prime_rl configs/opd-gap; do
  rsync -az --exclude .git -e "$ssh_cmd" "$source_repo/$path/" "$remote:$repo/$path/"
done
rsync -az -e "$ssh_cmd" "$source_repo/scripts/opd_gap_audit_diag_topk.py" "$remote:$repo/scripts/"

write_state dryrun "$config"
ssh "${ssh_args[@]}" "$remote" \
  "cd '$repo' && rm -rf /tmp/r40-gepaplan-dry && ~/.local/bin/uv run --no-sync rl @ '$config' --dry-run --output-dir /tmp/r40-gepaplan-dry"

write_state running "$output"
ssh "${ssh_args[@]}" "$remote" \
  "cd '$repo' && set -a && source .env && set +a && export PATH=\"$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin\" && ~/.local/bin/uv run --no-sync rl @ '$config'" \
  2>&1 | tee "$log"

write_state auditing "$output"
for step in 0 1; do
  ssh "${ssh_args[@]}" "$remote" \
    "cd '$repo' && ~/.local/bin/uv run --no-sync scripts/opd_gap_audit_diag_topk.py '$output/token_exports/step_${step}'" \
    | tee "/home/ubuntu/opd-gap-r40-gepaplan-audit-step${step}.json"
done

write_state complete "two optimizer steps and both field audits passed"
trap - ERR
