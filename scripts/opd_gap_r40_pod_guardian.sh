#!/usr/bin/env bash

set -euo pipefail

pod_id=${R40_POD_ID:-e87271e0b59843ce88a7181a5ce75c69}
controller=${R40_CONTROLLER:-opd-gap-r40-smoke-controller.service}
state_file=${R40_STATE_FILE:-/home/ubuntu/opd-gap-r40-smoke-controller-state.json}
guardian_state=${R40_GUARDIAN_STATE:-/home/ubuntu/opd-gap-r40-pod-guardian-state.json}
key=${R40_SSH_KEY:-/home/ubuntu/.ssh/primeintellect_ed25519}
prime=${R40_PRIME:-/home/ubuntu/.local/bin/prime}
artifact_root=${R40_ARTIFACT_ROOT:-/home/ubuntu/opd-gap-artifacts/r40-expanded-band-smokes}
max_seconds=${R40_MAX_SECONDS:-21600}
started=$(date +%s)

write_state() {
  jq -n --arg phase "$1" --arg detail "${2:-}" --arg pod_id "$pod_id" \
    '{phase:$phase,detail:$detail,pod_id:$pod_id,updated_at_utc:(now|todateiso8601)}' \
    >"$guardian_state.tmp"
  mv "$guardian_state.tmp" "$guardian_state"
}

pod_ssh() {
  env -u PRIME_API_KEY -u PRIME_TEAM_ID "$prime" pods status "$pod_id" --plain 2>/dev/null \
    | sed -n 's/^SSH[[:space:]]*//p' | xargs
}

terminate_pod() {
  env -u PRIME_API_KEY -u PRIME_TEAM_ID "$prime" pods terminate "$pod_id" --yes || true
}

preserve_artifacts() {
  local ssh_line remote port ssh_cmd repo output arm
  ssh_line=$(pod_ssh)
  [[ -n "$ssh_line" && "$ssh_line" != "N/A" ]] || return 0
  remote=${ssh_line%% *}
  port=$(awk '{for (i=1;i<=NF;i++) if ($i=="-p") print $(i+1)}' <<<"$ssh_line")
  port=${port:-22}
  ssh_cmd="ssh -i $key -p $port -o StrictHostKeyChecking=no"
  repo=/home/ubuntu/prime-rl
  mkdir -p "$artifact_root"
  for arm in fullanswer answerplan; do
    output="outputs-genagent-opsd-1lp-d64-${arm}-band000060-k8-tp4-pod-r40-smoke2-20260715"
    mkdir -p "$artifact_root/$arm"
    rsync -az --partial -e "$ssh_cmd" \
      --exclude 'run_default/checkpoints/***' \
      --exclude 'run_default/broadcasts/***' \
      --exclude '*.safetensors' \
      "$remote:$repo/$output/" "$artifact_root/$arm/" || true
  done
  cp -f /home/ubuntu/opd-gap-r40-*-smoke.log "$artifact_root/" 2>/dev/null || true
  cp -f /home/ubuntu/opd-gap-r40-*-audit-step*.json "$artifact_root/" 2>/dev/null || true
  cp -f /home/ubuntu/opd-gap-r40-smoke-controller{.log,-state.json} "$artifact_root/" 2>/dev/null || true
}

submit_full_runs() {
  local output job_ids
  output=$(ssh ar 'set -e; cd ~/prime-rl; \
    .venv/bin/rl @ configs/opd-gap/genagent-band000060-qwen35-opsd-fullanswer-ar-r40-full100.toml; \
    .venv/bin/rl @ configs/opd-gap/genagent-band000060-qwen35-opsd-answerplan-ar-r40-full100.toml')
  printf '%s\n' "$output" >"$artifact_root/full-submit.log"
  job_ids=$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"$output")
  [[ $(wc -w <<<"$job_ids") -eq 2 ]]
  for job_id in $job_ids; do
    ssh ar "scontrol update JobId=$job_id Nice=10000"
  done
  printf '%s\n' $job_ids >"$artifact_root/full-job-ids.txt"
}

write_state monitoring "waiting for two real smoke audits"
while systemctl --user is-active --quiet "$controller"; do
  now=$(date +%s)
  phase=$(jq -r '.phase // "unknown"' "$state_file" 2>/dev/null || printf unknown)
  if [[ "$phase" == "fullanswer_running" || "$phase" == "fullanswer_auditing" || "$phase" == answerplan_* ]]; then
    ssh ar 'scancel 228 229 2>/dev/null || true'
  fi
  if (( now - started > max_seconds )); then
    write_state failed "controller exceeded ${max_seconds}s hard ceiling"
    systemctl --user stop "$controller" || true
    preserve_artifacts
    terminate_pod
    exit 1
  fi
  write_state monitoring "controller phase: $phase"
  sleep 60
done

phase=$(jq -r '.phase // "unknown"' "$state_file" 2>/dev/null || printf unknown)
preserve_artifacts
if [[ "$phase" == "complete" ]]; then
  for arm in fullanswer answerplan; do
    for step in 0 1; do
      test -s "/home/ubuntu/opd-gap-r40-${arm}-audit-step${step}.json"
    done
  done
  write_state promoting "both smoke audits passed; submitting matched full runs"
  submit_full_runs
  terminate_pod
  write_state complete "artifacts preserved, full runs submitted, pod terminated"
  exit 0
fi

write_state failed "controller ended in phase: $phase"
terminate_pod
exit 1
