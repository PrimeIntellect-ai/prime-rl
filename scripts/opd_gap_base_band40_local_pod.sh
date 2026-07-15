#!/usr/bin/env bash

set -euo pipefail

pod_id=${1:?usage: $0 POD_ID}
controller_repo=/home/ubuntu/prime-rl
task_ref=a2b76f6ac3469f7f50171760c0d0dba38360edc4
ssh_key=/home/ubuntu/.ssh/primeintellect_ed25519
artifact_dir=/home/ubuntu/opd-gap-artifacts/genagent-base-band40-k32-local-r01
state_file=/home/ubuntu/opd-gap-base-band40-local-state.json
log_file=/home/ubuntu/opd-gap-base-band40-local-orchestrator.log
expected_model_revision=59d61f3ce65a6d9863b86d2e96597125219dc754

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$log_file"
}

prime_cli() {
  PRIME_API_KEY="$(jq -r .api_key /home/ubuntu/.prime/config.json)" \
    PRIME_TEAM_ID="$(jq -r .team_id /home/ubuntu/.prime/config.json)" \
    /home/ubuntu/.local/bin/prime "$@"
}

write_state() {
  local phase=$1
  local detail=${2:-}
  jq -n \
    --arg pod_id "$pod_id" \
    --arg phase "$phase" \
    --arg detail "$detail" \
    '{pod_id:$pod_id,phase:$phase,detail:$detail,updated_at_utc:(now|todateiso8601)}' \
    >"$state_file.tmp"
  mv "$state_file.tmp" "$state_file"
}

retry() {
  local attempts=$1
  shift
  local n=1
  until "$@"; do
    if (( n >= attempts )); then
      return 1
    fi
    log "retry $n/$attempts failed: $*"
    n=$((n + 1))
    sleep 15
  done
}

status_json=
for _ in $(seq 1 120); do
  status_json=$(prime_cli pods status "$pod_id" --output json --plain 2>>"$log_file" || true)
  status=$(jq -r '.status // empty' <<<"$status_json" 2>/dev/null || true)
  write_state provisioning "${status:-status unavailable}"
  if [[ "$status" == "ACTIVE" ]]; then
    break
  fi
  if [[ "$status" =~ ^(TERMINATED|FAILED|ERROR)$ ]]; then
    log "pod entered terminal state before bootstrap: $status"
    write_state failed "$status"
    exit 1
  fi
  sleep 30
done

if [[ $(jq -r '.status // empty' <<<"$status_json") != "ACTIVE" ]]; then
  log "pod did not become ACTIVE within 60 minutes"
  prime_cli pods terminate "$pod_id" --yes --plain >>"$log_file" 2>&1 || true
  write_state failed_terminated provisioning_timeout
  exit 1
fi

ssh_desc=$(jq -r '.ssh' <<<"$status_json")
user_host=${ssh_desc%% *}
port=$(sed -nE 's/.* -p ([0-9]+).*/\1/p' <<<"$ssh_desc")
port=${port:-22}
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -i "$ssh_key" -p "$port")
remote_ssh=(ssh "${ssh_opts[@]}" "$user_host")
remote_rsync="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new -i $ssh_key -p $port"

log "pod active endpoint=$user_host port=$port"
write_state ssh_wait "$user_host:$port"
for _ in $(seq 1 60); do
  if "${remote_ssh[@]}" true >/dev/null 2>&1; then
    break
  fi
  sleep 10
done
"${remote_ssh[@]}" true
remote_home=$("${remote_ssh[@]}" 'printf %s "$HOME"')
remote_repo="$remote_home/prime-rl"
remote_smoke="$remote_repo/evals/genagent-base-band40-k8-local-smoke-r01-20260715"
remote_full="$remote_repo/evals/genagent-base-band40-k32-local-r01-20260715"

cleanup() {
  local rc=$?
  if (( rc != 0 )); then
    log "local screen failed rc=$rc; collecting reachable artifacts and terminating pod"
    mkdir -p "$artifact_dir"
    "${remote_ssh[@]}" "pkill -f '/inference @.*qwen35-base-band-screen-tp4.toml' || true" >/dev/null 2>&1 || true
    rsync -az -e "$remote_rsync" "$user_host:$remote_smoke/" "$artifact_dir/smoke/" >>"$log_file" 2>&1 || true
    rsync -az -e "$remote_rsync" "$user_host:$remote_full/" "$artifact_dir/full/" >>"$log_file" 2>&1 || true
    rsync -az -e "$remote_rsync" "$user_host:$remote_home/opd-gap-base-band40-inference.log" "$artifact_dir/" >>"$log_file" 2>&1 || true
    prime_cli pods terminate "$pod_id" --yes --plain >>"$log_file" 2>&1 || true
    write_state failed_terminated "exit $rc"
  fi
}
trap cleanup EXIT

write_state bootstrap "$user_host:$port"
log "installing base utilities"
retry 3 "${remote_ssh[@]}" 'if command -v sudo >/dev/null; then sudo -n apt-get update -qq && sudo -n apt-get install -y -qq rsync git curl jq build-essential ninja-build; else apt-get update -qq && apt-get install -y -qq rsync git curl jq build-essential ninja-build; fi'

log "syncing branch-local code and exact General Agent task cache"
"${remote_ssh[@]}" "mkdir -p '$remote_repo' '$remote_home/.cache/verifiers/general_agent/$task_ref'"
retry 3 rsync -az -e "$remote_rsync" \
  --exclude .git --exclude .venv --exclude evals --exclude 'outputs-*' --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache \
  "$controller_repo/" "$user_host:$remote_repo/"
retry 3 rsync -az -e "$remote_rsync" "$controller_repo/.env" "$user_host:$remote_repo/.env"
retry 3 rsync -az -e "$remote_rsync" \
  "/home/ubuntu/.cache/verifiers/general_agent/$task_ref/" \
  "$user_host:$remote_home/.cache/verifiers/general_agent/$task_ref/"

log "installing frozen Python environment"
retry 3 "${remote_ssh[@]}" "bash -lc 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=\"\$HOME/.local/bin:\$PATH\"; cd \"$remote_repo\"; uv sync --frozen --extra flash-attn --extra envs --inexact'"
gpu_report=$("${remote_ssh[@]}" "bash -lc 'cd \"$remote_repo\"; .venv/bin/python -c \"import torch, vllm; print(torch.cuda.device_count(), vllm.__version__)\"'")
log "runtime report: $gpu_report"
[[ $gpu_report == 4\ * ]]

write_state serving base_model
log "launching local TP4 vLLM server"
"${remote_ssh[@]}" "bash -lc 'set -e; cd \"$remote_repo\"; set -a; source .env; set +a; export HF_HOME=\"\$HOME/.cache/huggingface\"; export TMPDIR=/tmp/opd-gap-base-band40; export UV_CACHE_DIR=\"\$TMPDIR/uv-cache\"; export XDG_CACHE_HOME=\"\$TMPDIR/xdg-cache\"; export TRITON_CACHE_DIR=\"\$TMPDIR/triton\"; export TORCHINDUCTOR_CACHE_DIR=\"\$TMPDIR/torchinductor\"; export VLLM_CACHE_ROOT=\"\$TMPDIR/vllm\"; mkdir -p \"\$UV_CACHE_DIR\" \"\$XDG_CACHE_HOME\" \"\$TRITON_CACHE_DIR\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$VLLM_CACHE_ROOT\"; ulimit -n 65536; nohup .venv/bin/inference @ configs/opd-gap/qualification/qwen35-base-band-screen-tp4.toml >\"$remote_home/opd-gap-base-band40-inference.log\" 2>&1 & echo \$! >\"$remote_home/opd-gap-base-band40-inference.pid\"'"

for _ in $(seq 1 180); do
  if "${remote_ssh[@]}" 'curl -fsS http://127.0.0.1:8000/v1/models >/dev/null'; then
    break
  fi
  if ! "${remote_ssh[@]}" "kill -0 \$(cat '$remote_home/opd-gap-base-band40-inference.pid') 2>/dev/null"; then
    log "vLLM exited before readiness"
    exit 1
  fi
  sleep 10
done
"${remote_ssh[@]}" 'curl -fsS http://127.0.0.1:8000/v1/models >/dev/null'
served_revision=$("${remote_ssh[@]}" "cat '$remote_home/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/refs/main' 2>/dev/null || true")
log "served model revision=${served_revision:-unknown}"
[[ "$served_revision" == "$expected_model_revision" ]]

write_state smoke "1 task x 8 rollouts"
log "running mandatory local 8-rollout smoke"
"${remote_ssh[@]}" "rm -f '$remote_home/opd-gap-base-band40-smoke.exit'; nohup bash -lc 'cd \"$remote_repo\"; export VLLM_API_KEY=EMPTY; .venv/bin/eval @ configs/opd-gap/qualification/genagent-base-band40-k32-local-r01.toml --num-tasks 1 --num-rollouts 8 --max-concurrent 8 --output-dir \"$remote_smoke\"; rc=\$?; printf %s \"\$rc\" >\"$remote_home/opd-gap-base-band40-smoke.exit\"; exit \"\$rc\"' >'$remote_home/opd-gap-base-band40-smoke.log' 2>&1 & echo \$! >'$remote_home/opd-gap-base-band40-smoke.pid'"
while ! smoke_rc=$("${remote_ssh[@]}" "cat '$remote_home/opd-gap-base-band40-smoke.exit' 2>/dev/null"); do
  if ! "${remote_ssh[@]}" "kill -0 \$(cat '$remote_home/opd-gap-base-band40-smoke.pid') 2>/dev/null"; then
    log "local smoke exited without an exit marker"
    exit 1
  fi
  write_state smoke "$("${remote_ssh[@]}" "wc -l <'$remote_smoke/results.jsonl' 2>/dev/null || echo 0")/8 rows"
  sleep 30
done
[[ "$smoke_rc" == "0" ]]
"${remote_ssh[@]}" "jq -se 'length == 8 and all(.is_completed == true and (.errors | length) == 0 and .stop_condition != \"error\")' '$remote_smoke/results.jsonl' >/dev/null"
log "smoke passed: 8/8 complete and error-free"

write_state evaluating "40 tasks x 32 rollouts"
log "running full local base-model screen"
"${remote_ssh[@]}" "rm -f '$remote_home/opd-gap-base-band40-eval.exit'; nohup bash -lc 'cd \"$remote_repo\"; export VLLM_API_KEY=EMPTY; .venv/bin/eval @ configs/opd-gap/qualification/genagent-base-band40-k32-local-r01.toml; rc=\$?; printf %s \"\$rc\" >\"$remote_home/opd-gap-base-band40-eval.exit\"; exit \"\$rc\"' >'$remote_home/opd-gap-base-band40-eval.log' 2>&1 & echo \$! >'$remote_home/opd-gap-base-band40-eval.pid'"
while ! eval_rc=$("${remote_ssh[@]}" "cat '$remote_home/opd-gap-base-band40-eval.exit' 2>/dev/null"); do
  if ! "${remote_ssh[@]}" "kill -0 \$(cat '$remote_home/opd-gap-base-band40-eval.pid') 2>/dev/null"; then
    log "local full screen exited without an exit marker"
    exit 1
  fi
  completed=$("${remote_ssh[@]}" "wc -l <'$remote_full/results.jsonl' 2>/dev/null || echo 0")
  errors=$("${remote_ssh[@]}" "jq -s 'map(select((.errors | length) > 0)) | length' '$remote_full/results.jsonl' 2>/dev/null || echo 0")
  write_state evaluating "$completed/1280 rows; $errors errors"
  if (( errors > 0 )); then
    log "local full screen produced an error row after retries"
    exit 1
  fi
  sleep 30
done
[[ "$eval_rc" == "0" ]]
"${remote_ssh[@]}" "jq -se 'length == 1280 and all(.is_completed == true and (.errors | length) == 0 and .stop_condition != \"error\")' '$remote_full/results.jsonl' >/dev/null"

write_state copying "validated 1280 rows"
log "copying validated local artifacts"
mkdir -p "$artifact_dir"
retry 3 rsync -az -e "$remote_rsync" "$user_host:$remote_smoke/" "$artifact_dir/smoke/"
retry 3 rsync -az -e "$remote_rsync" "$user_host:$remote_full/" "$artifact_dir/full/"
retry 3 rsync -az -e "$remote_rsync" "$user_host:$remote_home/opd-gap-base-band40-inference.log" "$artifact_dir/"
retry 3 rsync -az -e "$remote_rsync" "$user_host:$remote_home/opd-gap-base-band40-eval.log" "$artifact_dir/"

log "terminating paid local-evaluation pod"
prime_cli pods terminate "$pod_id" --yes --plain >>"$log_file" 2>&1
trap - EXIT
write_state complete_terminated "$artifact_dir/full"
log "complete: 1280 local rows copied and pod terminated"
