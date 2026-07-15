#!/usr/bin/env bash

set -euo pipefail

pod_id=${R41_POD_ID:?set R41_POD_ID}
remote=${R41_REMOTE:?set R41_REMOTE, for example ubuntu@host}
port=${R41_PORT:-22}
key=${R41_SSH_KEY:-/home/ubuntu/.ssh/primeintellect_ed25519}
prime=${R41_PRIME:-/home/ubuntu/.local/bin/prime}
repo=${R41_REMOTE_REPO:-/home/ubuntu/prime-rl}
uv=${R41_REMOTE_UV:-/home/ubuntu/.local/bin/uv}
state=${R41_STATE:-/home/ubuntu/opd-gap-r41-resume-smoke-controller-state.json}
log=${R41_LOG:-/home/ubuntu/opd-gap-r41-resume-smoke-controller.log}
artifact_root=${R41_ARTIFACT_ROOT:-/home/ubuntu/opd-gap-artifacts/r41-exact-band-smokes-resume}
old_controller_pid=${R41_OLD_CONTROLLER_PID:-}

ssh_args=(-i "$key" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
ssh_cmd="ssh -i $key -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

exec >>"$log" 2>&1

write_state() {
  jq -n --arg phase "$1" --arg detail "${2:-}" --arg pod_id "$pod_id" \
    '{phase:$phase,detail:$detail,pod_id:$pod_id,updated_at_utc:(now|todateiso8601)}' \
    >"$state.tmp"
  mv "$state.tmp" "$state"
}

preserve_artifacts() {
  mkdir -p "$artifact_root"
  for arm in fullanswer answerplan; do
    local output="outputs-genagent-opsd-1lp-d64-${arm}-band000060-k8-tp4-pod-r41-smoke2-20260715"
    mkdir -p "$artifact_root/$arm"
    rsync -az --partial -e "$ssh_cmd" \
      --exclude 'run_default/checkpoints/***' \
      --exclude 'run_default/broadcasts/***' \
      --exclude '*.safetensors' \
      "$remote:$repo/$output/" "$artifact_root/$arm/" || true
  done
  cp -f /home/ubuntu/opd-gap-r41-*-audit-step*.json "$artifact_root/" 2>/dev/null || true
  cp -f /home/ubuntu/opd-gap-r41-*-policy-step*.json "$artifact_root/" 2>/dev/null || true
  cp -f /home/ubuntu/opd-gap-r41-*-rollout-step*.json "$artifact_root/" 2>/dev/null || true
  cp -f "$log" "$state" "$artifact_root/" 2>/dev/null || true
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  preserve_artifacts
  env -u PRIME_API_KEY -u PRIME_TEAM_ID "$prime" pods terminate "$pod_id" --yes || true
  if (( code != 0 )); then
    write_state failed "resume controller exit $code; artifacts preserved and pod termination requested"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

audit_arm() {
  local arm=$1
  local output="outputs-genagent-opsd-1lp-d64-${arm}-band000060-k8-tp4-pod-r41-smoke2-20260715"
  local failed=0
  write_state "${arm}_auditing" "$output"
  for step in 0 1; do
    local rollouts="$output/run_default/rollouts/step_${step}/train_rollouts.jsonl"
    ssh "${ssh_args[@]}" "$remote" \
      "cd '$repo' && jq -s '{rows:length,error_rows:([.[]|select((.errors|length)>0)]|length),agent_completed:([.[]|select(.stop_condition==\"agent_completed\")]|length),harness_timeouts:([.[]|select(.stop_condition==\"harness_timeout\")]|length)}' '$rollouts'" \
      | tee "/home/ubuntu/opd-gap-r41-${arm}-rollout-step${step}.json" || failed=1
    ssh "${ssh_args[@]}" "$remote" \
      "cd '$repo' && '$uv' run --no-sync scripts/opd_gap_audit_policy_step.py '$output' '$step'" \
      | tee "/home/ubuntu/opd-gap-r41-${arm}-policy-step${step}.json" || failed=1
    ssh "${ssh_args[@]}" "$remote" \
      "cd '$repo' && '$uv' run --no-sync scripts/opd_gap_audit_diag_topk.py '$output/run_default/token_exports/step_${step}'" \
      | tee "/home/ubuntu/opd-gap-r41-${arm}-audit-step${step}.json" || failed=1
  done
  preserve_artifacts
  return "$failed"
}

fullanswer_config=configs/opd-gap/genagent-band000060-qwen35-opsd-fullanswer-pod-r41-smoke2.toml
write_state fullanswer_waiting "adopted already-running full-answer smoke"
for _ in $(seq 1 1080); do
  if ! ssh "${ssh_args[@]}" "$remote" "pgrep -f '[r]l @ $fullanswer_config' >/dev/null"; then
    break
  fi
  sleep 10
done
if ssh "${ssh_args[@]}" "$remote" "pgrep -f '[r]l @ $fullanswer_config' >/dev/null"; then
  write_state failed "full-answer smoke exceeded three hours"
  exit 1
fi

if [[ -n "$old_controller_pid" ]]; then
  kill -KILL "$old_controller_pid" 2>/dev/null || true
fi
fullanswer_audit=0
audit_arm fullanswer || fullanswer_audit=$?

answerplan_config=configs/opd-gap/genagent-band000060-qwen35-opsd-answerplan-pod-r41-smoke2.toml
write_state answerplan_dryrun "$answerplan_config"
ssh "${ssh_args[@]}" "$remote" \
  "cd '$repo' && rm -rf /tmp/r41-answerplan-resume-dry && '$uv' run --no-sync rl @ '$answerplan_config' --dry-run --output-dir /tmp/r41-answerplan-resume-dry"
write_state answerplan_running "$answerplan_config"
timeout --signal=TERM --kill-after=5m 3h \
  ssh "${ssh_args[@]}" "$remote" \
    "cd '$repo' && set -a && source .env && set +a && export PATH='/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin' && '$uv' run --no-sync rl @ '$answerplan_config'"
answerplan_audit=0
audit_arm answerplan || answerplan_audit=$?

if (( fullanswer_audit != 0 || answerplan_audit != 0 )); then
  write_state failed "mechanics artifacts preserved, but one or more strict completion/field audits failed"
  exit 1
fi

write_state complete "four resampling-aware field audits passed; artifacts preserved; pod termination requested"
