#!/usr/bin/env bash

set -euo pipefail

state=${R41_AR_STATE:-/home/ubuntu/opd-gap-r41-ar-monitor-state.json}
artifact_root=${R41_AR_ARTIFACT_ROOT:-/home/ubuntu/opd-gap-artifacts/r41-ar}
repo=/home/tim/prime-rl

# Full-run promotion is an explicit operator action. A resurrected service must
# fail closed unless the operator opts in on that exact invocation.
if [[ ${R41_ALLOW_AUTO_PROMOTION:-0} != 1 ]]; then
  mkdir -p "$(dirname "$state")"
  jq -n \
    --arg phase held \
    --arg detail "automatic full-run promotion disabled; set R41_ALLOW_AUTO_PROMOTION=1 explicitly to enable" \
    '{phase:$phase,detail:$detail,updated_at_utc:(now|todateiso8601)}' \
    >"$state.tmp"
  mv "$state.tmp" "$state"
  exit 0
fi

mkdir -p "$artifact_root/audits"

declare -A smoke_jobs=(
  [grpo]=231
  [fullanswer]=232
  [answerplan]=234
)
declare -A smoke_outputs=(
  [grpo]=outputs-genagent-grpo-band000060-k8-tp4-ar-r41-smoke2-20260715
  [fullanswer]=outputs-genagent-opsd-1lp-d64-fullanswer-band000060-k8-tp4-ar-r41-smoke2-20260715
  [answerplan]=outputs-genagent-opsd-1lp-d64-answerplan-band000060-k8-tp4-ar-r41-smoke2-20260715
)
declare -A full_configs=(
  [grpo]=configs/opd-gap/genagent-band000060-qwen35-grpo-ar-r41-full100.toml
  [fullanswer]=configs/opd-gap/genagent-band000060-qwen35-opsd-fullanswer-ar-r41-full100.toml
  [answerplan]=configs/opd-gap/genagent-band000060-qwen35-opsd-answerplan-ar-r41-full100.toml
)
declare -A full_outputs=(
  [grpo]=outputs-genagent-grpo-band000060-k8-tp4-ar-r41-full100-20260715
  [fullanswer]=outputs-genagent-opsd-1lp-d64-fullanswer-band000060-k8-tp4-ar-r41-full100-20260715
  [answerplan]=outputs-genagent-opsd-1lp-d64-answerplan-band000060-k8-tp4-ar-r41-full100-20260715
)

write_state() {
  local phase=$1 detail=$2
  jq -n --arg phase "$phase" --arg detail "$detail" \
    --arg grpo_smoke "${smoke_jobs[grpo]}" \
    --arg fullanswer_smoke "${smoke_jobs[fullanswer]}" \
    --arg answerplan_smoke "${smoke_jobs[answerplan]}" \
    --argjson full_jobs "$(cat "$artifact_root/full-job-ids.json" 2>/dev/null || printf '{}')" \
    '{phase:$phase,detail:$detail,smoke_jobs:{grpo:$grpo_smoke,fullanswer:$fullanswer_smoke,answerplan:$answerplan_smoke},full_jobs:$full_jobs,updated_at_utc:(now|todateiso8601)}' \
    >"$state.tmp"
  mv "$state.tmp" "$state"
}

remote_file() {
  ssh ar "test -f '$repo/$1'"
}

job_state() {
  ssh ar "squeue -h -j '$1' -o %T" 2>/dev/null | head -1
}

runtime_seconds() {
  local raw days=0 h=0 m=0 s=0
  raw=$(ssh ar "squeue -h -j '$1' -o %M" 2>/dev/null | head -1)
  [[ -n "$raw" ]] || { printf '0\n'; return; }
  if [[ "$raw" == *-* ]]; then
    days=${raw%%-*}
    raw=${raw#*-}
  fi
  IFS=: read -r h m s <<<"$raw"
  if [[ -z "${s:-}" ]]; then
    s=$m
    m=$h
    h=0
  fi
  printf '%s\n' $((10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s))
}

audit_opsd_step() {
  local arm=$1 output=$2 step=$3 audit="$artifact_root/audits/${arm}-step${step}.json"
  [[ -s "$audit" ]] && return 0
  ssh ar "cd '$repo' && .venv/bin/python scripts/opd_gap_audit_diag_topk.py '$output/run_default/token_exports/step_${step}'" >"$audit.tmp"
  mv "$audit.tmp" "$audit"
}

audit_policy_step() {
  local arm=$1 output=$2 step=$3 audit="$artifact_root/audits/${arm}-policy-step${step}.json"
  local extra=()
  [[ -s "$audit" ]] && return 0
  if [[ "$arm" == grpo || "$arm" == grpo-full ]]; then
    extra=(--require-reward-variance --require-nonzero-advantages)
  fi
  ssh ar "cd '$repo' && .venv/bin/python scripts/opd_gap_audit_policy_step.py '$output' '$step' ${extra[*]}" >"$audit.tmp"
  mv "$audit.tmp" "$audit"
}

smoke_ready() {
  local arm=$1 output=${smoke_outputs[$1]}
  remote_file "$output/run_default/token_exports/step_0/STABLE" || return 1
  remote_file "$output/run_default/token_exports/step_1/STABLE" || return 1
  audit_policy_step "$arm" "$output" 0
  audit_policy_step "$arm" "$output" 1
  if [[ "$arm" != grpo ]]; then
    audit_opsd_step "$arm" "$output" 0
    audit_opsd_step "$arm" "$output" 1
  fi
}

submit_full() {
  local arm=$1 marker="$artifact_root/${arm}-full-job-id"
  [[ -s "$marker" ]] && return 0
  local output job_id
  output=$(ssh ar "cd '$repo' && .venv/bin/rl @ '${full_configs[$arm]}'")
  job_id=$(sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' <<<"$output" | tail -1)
  [[ -n "$job_id" ]]
  printf '%s\n' "$job_id" >"$marker"
}

refresh_full_json() {
  jq -n \
    --arg grpo "$(cat "$artifact_root/grpo-full-job-id" 2>/dev/null || true)" \
    --arg fullanswer "$(cat "$artifact_root/fullanswer-full-job-id" 2>/dev/null || true)" \
    --arg answerplan "$(cat "$artifact_root/answerplan-full-job-id" 2>/dev/null || true)" \
    '{grpo:$grpo,fullanswer:$fullanswer,answerplan:$answerplan}' \
    >"$artifact_root/full-job-ids.json"
}

latest_stable_step() {
  local output=$1
  ssh ar "find '$repo/$output/run_default/token_exports' -mindepth 2 -maxdepth 2 -type f -name STABLE 2>/dev/null | sed -n 's#.*step_\([0-9][0-9]*\)/STABLE#\1#p' | sort -n | tail -1" 2>/dev/null
}

audit_full_milestones() {
  local arm=$1 output=${full_outputs[$1]} latest=$2 step
  for step in 0 10 20 30 40 50 60 70 80 90 99; do
    (( step <= latest )) || continue
    audit_policy_step "$arm-full" "$output" "$step"
    if [[ "$arm" != grpo ]]; then
      audit_opsd_step "$arm-full" "$output" "$step"
    fi
  done
}

write_state queued "waiting for r41 AR smoke jobs"
while true; do
  all_smokes=true
  for arm in grpo fullanswer answerplan; do
    if smoke_ready "$arm"; then
      continue
    fi
    all_smokes=false
    state_now=$(job_state "${smoke_jobs[$arm]}")
    if [[ "$state_now" == RUNNING ]] && (( $(runtime_seconds "${smoke_jobs[$arm]}") > 5400 )); then
      ssh ar "scancel '${smoke_jobs[$arm]}'"
      write_state failed "$arm smoke exceeded 90 minutes without two stable steps"
      exit 1
    fi
    if [[ -z "$state_now" ]]; then
      write_state failed "$arm smoke left Slurm without two stable exports"
      exit 1
    fi
  done

  if [[ "$all_smokes" == true ]]; then
    for arm in grpo fullanswer answerplan; do submit_full "$arm"; done
    refresh_full_json
  fi

  if [[ -s "$artifact_root/full-job-ids.json" ]]; then
    all_full=true
    for arm in grpo fullanswer answerplan; do
      job_id=$(jq -r --arg arm "$arm" '.[$arm]' "$artifact_root/full-job-ids.json")
      [[ -n "$job_id" ]] || { all_full=false; continue; }
      latest=$(latest_stable_step "${full_outputs[$arm]}")
      latest=${latest:--1}
      audit_full_milestones "$arm" "${full_outputs[$arm]}" "$latest"
      if (( latest >= 99 )); then
        continue
      fi
      all_full=false
      state_now=$(job_state "$job_id")
      if [[ -z "$state_now" ]]; then
        write_state failed "$arm full run left Slurm at stable step $latest"
        exit 1
      fi
      if [[ "$state_now" == RUNNING && "$latest" -lt 0 ]] && (( $(runtime_seconds "$job_id") > 5400 )); then
        ssh ar "scancel '$job_id'"
        write_state failed "$arm full run exceeded 90 minutes without step 0"
        exit 1
      fi
    done
    if [[ "$all_full" == true ]]; then
      write_state complete "all three r41 runs reached stable step 99 and OPSD milestone audits passed"
      exit 0
    fi
  fi

  write_state monitoring "smokes/full runs queued or active"
  sleep 120
done
