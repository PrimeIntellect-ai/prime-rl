#!/usr/bin/env bash

set -euo pipefail

pod_id=${R41_POD_ID:?set R41_POD_ID to the allocated smoke pod}
prime=${R41_PRIME:-/home/ubuntu/.local/bin/prime}
key=${R41_SSH_KEY:-/home/ubuntu/.ssh/primeintellect_ed25519}
source_repo=${R41_SOURCE_REPO:-/home/ubuntu/prime-rl}
state=${R41_STATE:-/home/ubuntu/opd-gap-r41-smoke-controller-state.json}
log=${R41_LOG:-/home/ubuntu/opd-gap-r41-smoke-controller.log}
artifact_root=${R41_ARTIFACT_ROOT:-/home/ubuntu/opd-gap-artifacts/r41-exact-band-smokes}
task_ref=a2b76f6ac3469f7f50171760c0d0dba38360edc4
repo=
remote_home=
remote=
port=22

exec >>"$log" 2>&1

renderers_version=$(cd "$source_repo/deps/renderers" && /home/ubuntu/.local/bin/uvx --from hatch hatch version)
verifiers_version=$(cd "$source_repo/deps/verifiers" && /home/ubuntu/.local/bin/uvx --from hatch hatch version)
[[ "$renderers_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]
[[ "$verifiers_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]

write_state() {
  jq -n --arg phase "$1" --arg detail "${2:-}" --arg pod_id "$pod_id" \
    '{phase:$phase,detail:$detail,pod_id:$pod_id,updated_at_utc:(now|todateiso8601)}' \
    >"$state.tmp"
  mv "$state.tmp" "$state"
}

pod_ssh() {
  env -u PRIME_API_KEY -u PRIME_TEAM_ID "$prime" pods status "$pod_id" --plain 2>/dev/null \
    | sed -n 's/^[[:space:]]*SSH[[:space:]]*//p' | xargs
}

preserve_artifacts() {
  [[ -n "$remote" && -n "$repo" ]] || return 0
  local ssh_cmd="ssh -i $key -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
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
  cp -f /home/ubuntu/opd-gap-r41-*-rollout-step*.json "$artifact_root/" 2>/dev/null || true
  cp -f "$log" "$state" "$artifact_root/" 2>/dev/null || true
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  preserve_artifacts
  env -u PRIME_API_KEY -u PRIME_TEAM_ID "$prime" pods terminate "$pod_id" --yes || true
  if (( code != 0 )); then
    write_state failed "controller exit $code; artifacts preserved and pod termination requested"
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

write_state provisioning "waiting for pod SSH"
for _ in $(seq 1 160); do
  ssh_line=$(pod_ssh)
  if [[ -n "$ssh_line" && "$ssh_line" != "N/A" ]]; then
    remote=${ssh_line%% *}
    port=$(awk '{for (i=1;i<=NF;i++) if ($i=="-p") print $(i+1)}' <<<"$ssh_line")
    port=${port:-22}
    break
  fi
  sleep 15
done
[[ -n "$remote" ]]

ssh_args=(-i "$key" -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
ssh_cmd="ssh -i $key -p $port -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
for _ in $(seq 1 40); do
  ssh "${ssh_args[@]}" "$remote" true 2>/dev/null && break
  sleep 15
done
ssh "${ssh_args[@]}" "$remote" true
remote_home=$(ssh "${ssh_args[@]}" "$remote" 'printf %s "$HOME"')
repo="$remote_home/prime-rl"
uv="$remote_home/.local/bin/uv"

write_state setup "installing tools and syncing exact r41 runtime"
ssh "${ssh_args[@]}" "$remote" \
  "if ! command -v git >/dev/null || ! command -v rsync >/dev/null || ! command -v curl >/dev/null || ! command -v g++ >/dev/null; then sudo apt-get update && sudo apt-get install -y git rsync curl g++; fi"
ssh "${ssh_args[@]}" "$remote" \
  "test -d '$repo/.git' || git clone --branch codex/opsd-lora-stability --single-branch https://github.com/PrimeIntellect-ai/prime-rl.git '$repo'"

for path in packages/prime-rl-configs src/prime_rl deps configs/opd-gap; do
  rsync -az --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
    -e "$ssh_cmd" "$source_repo/$path/" "$remote:$repo/$path/"
done
ssh "${ssh_args[@]}" "$remote" \
  "sed -i 's/^fallback-version = .*/fallback-version = \"$renderers_version\"/' '$repo/deps/renderers/pyproject.toml' && sed -i 's/^fallback-version = .*/fallback-version = \"$verifiers_version\"/' '$repo/deps/verifiers/pyproject.toml' && grep -qx 'fallback-version = \"$renderers_version\"' '$repo/deps/renderers/pyproject.toml' && grep -qx 'fallback-version = \"$verifiers_version\"' '$repo/deps/verifiers/pyproject.toml'"
rsync -az -e "$ssh_cmd" \
  "$source_repo/pyproject.toml" "$source_repo/uv.lock" \
  "$source_repo/scripts/opd_gap_audit_diag_topk.py" \
  "$source_repo/scripts/opd_gap_audit_policy_step.py" "$remote:$repo/scripts/"
if [[ -f "$source_repo/.env" ]]; then
  scp -q -i "$key" -P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$source_repo/.env" "$remote:$repo/.env"
fi
if [[ -f /home/ubuntu/.netrc ]]; then
  scp -q -i "$key" -P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /home/ubuntu/.netrc "$remote:$remote_home/.netrc"
  ssh "${ssh_args[@]}" "$remote" "chmod 600 '$remote_home/.netrc'"
fi

ssh "${ssh_args[@]}" "$remote" "mkdir -p '$remote_home/.cache/verifiers/general_agent/$task_ref'"
rsync -az -e "$ssh_cmd" \
  "/home/ubuntu/.cache/verifiers/general_agent_verified/$task_ref/" \
  "$remote:$remote_home/.cache/verifiers/general_agent/$task_ref/"

ssh "${ssh_args[@]}" "$remote" \
  "sudo mkdir -p '$remote_home/.config/uv' '$remote_home/.local/bin' && sudo chown -R \$(id -u):\$(id -g) '$remote_home/.config' '$remote_home/.local'"
ssh "${ssh_args[@]}" "$remote" \
  "test -x '$uv' || curl -LsSf https://astral.sh/uv/install.sh | XDG_CONFIG_HOME=/tmp/uv-config UV_NO_MODIFY_PATH=1 sh"
ssh "${ssh_args[@]}" "$remote" "cd '$repo' && '$uv' sync --frozen --all-extras"
ssh "${ssh_args[@]}" "$remote" \
  "cd '$repo' && '$uv' run --no-sync python -c 'import torch, verifiers; assert torch.cuda.device_count() == 8; print(torch.cuda.device_count())'"

run_arm() {
  local arm=$1
  local config="configs/opd-gap/genagent-band000060-qwen35-opsd-${arm}-pod-r41-smoke2.toml"
  local output="outputs-genagent-opsd-1lp-d64-${arm}-band000060-k8-tp4-pod-r41-smoke2-20260715"
  write_state "${arm}_dryrun" "$config"
  ssh "${ssh_args[@]}" "$remote" \
    "cd '$repo' && rm -rf '/tmp/r41-${arm}-dry' && '$uv' run --no-sync rl @ '$config' --dry-run --output-dir '/tmp/r41-${arm}-dry'"
  write_state "${arm}_running" "$output"
  timeout --signal=TERM --kill-after=5m 3h \
    ssh "${ssh_args[@]}" "$remote" \
      "cd '$repo' && set -a && source .env && set +a && export PATH='$remote_home/.local/bin:/usr/local/bin:/usr/bin:/bin' && '$uv' run --no-sync rl @ '$config'"
  write_state "${arm}_auditing" "$output"
  for step in 0 1; do
    local rollouts="$output/run_default/rollouts/step_${step}/train_rollouts.jsonl"
    ssh "${ssh_args[@]}" "$remote" \
      "cd '$repo' && jq -s '{rows:length,error_rows:([.[]|select((.errors|length)>0)]|length),agent_completed:([.[]|select(.stop_condition==\"agent_completed\")]|length),harness_timeouts:([.[]|select(.stop_condition==\"harness_timeout\")]|length)}' '$rollouts'" \
      | tee "/home/ubuntu/opd-gap-r41-${arm}-rollout-step${step}.json"
    ssh "${ssh_args[@]}" "$remote" \
      "cd '$repo' && '$uv' run --no-sync scripts/opd_gap_audit_policy_step.py '$output' '$step'" \
      | tee "/home/ubuntu/opd-gap-r41-${arm}-policy-step${step}.json"
    ssh "${ssh_args[@]}" "$remote" \
      "cd '$repo' && '$uv' run --no-sync scripts/opd_gap_audit_diag_topk.py '$output/run_default/token_exports/step_${step}'" \
      | tee "/home/ubuntu/opd-gap-r41-${arm}-audit-step${step}.json"
  done
  preserve_artifacts
}

run_arm fullanswer
run_arm answerplan
write_state complete "four field audits passed; artifacts preserved; pod termination requested"
