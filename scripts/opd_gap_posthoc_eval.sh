#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <arm-name> <training-output-dir> [include-base]" >&2
  exit 2
fi

arm_name=$1
training_output_dir=$2
include_base=${3:-false}
port=${POSTHOC_PORT:-8000}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="$repo_root/.venv/bin:$PATH"
training_output_dir=$(realpath "$training_output_dir")
eval_root="$training_output_dir/posthoc_evals"
inference_log="$eval_root/inference.log"
eval_config=${POSTHOC_EVAL_CONFIG:-"$repo_root/configs/opd-gap/posthoc/genagent-heldout-r28.toml"}
inference_config="$repo_root/configs/opd-gap/posthoc/qwen35-lora-inference.toml"
base_model="Qwen/Qwen3.5-35B-A3B"
expected_rows=${POSTHOC_EXPECTED_ROWS:-20}
posthoc_steps=${POSTHOC_STEPS:-"10 20 30 40 50 60 70 80 90 100"}
posthoc_steps=${posthoc_steps//,/ }

mkdir -p "$eval_root"
export VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}

has_snapshot=false
for step in $posthoc_steps; do
  if [[ -f "$training_output_dir/run_default/broadcasts/step_$step/STABLE" ]]; then
    has_snapshot=true
    break
  fi
done

if [[ "$include_base" != "true" && "$has_snapshot" != "true" ]]; then
  echo "No stable 10-step snapshots were available for $arm_name"
  exit 0
fi

cleanup() {
  if [[ -n "${inference_pid:-}" ]]; then
    kill "$inference_pid" 2>/dev/null || true
    wait "$inference_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$repo_root"
.venv/bin/inference @ "$inference_config" --server.port "$port" >"$inference_log" 2>&1 &
inference_pid=$!

for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$inference_pid" 2>/dev/null; then
    echo "inference server exited before becoming healthy; see $inference_log" >&2
    exit 1
  fi
  sleep 10
done

if ! curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
  echo "inference server did not become healthy within 30 minutes; see $inference_log" >&2
  exit 1
fi

run_eval() {
  local label=$1
  local model=$2
  local policy_step=$3
  local adapter_path=$4
  local output_dir="$eval_root/$label"

  if [[ -f "$output_dir/COMPLETE" ]]; then
    echo "Skipping completed evaluation $arm_name/$label"
    return
  fi

  mkdir -p "$output_dir"
  echo "Starting held-out evaluation $arm_name/$label model=$model"
  if [[ -f "$output_dir/config.toml" ]]; then
    .venv/bin/eval --resume "$output_dir"
  else
    .venv/bin/eval @ "$eval_config" \
      --model "$model" \
      --client.base-url "http://localhost:$port/v1" \
      --output-dir "$output_dir" \
      --dry-run false
  fi
  "$repo_root/scripts/opd_gap_summarize_eval.py" "$output_dir/results.jsonl" \
    --expected-rows "$expected_rows" \
    --policy-step "$policy_step" \
    --adapter-name "$model" \
    --adapter-path "$adapter_path" \
    --output "$output_dir/summary.json"
  touch "$output_dir/COMPLETE"
}

if [[ "$include_base" == "true" ]]; then
  run_eval step_0 "$base_model" 0 "$base_model"
fi

evaluated=0
for step in $posthoc_steps; do
  adapter_dir="$training_output_dir/run_default/broadcasts/step_$step"
  if [[ ! -f "$adapter_dir/STABLE" ]]; then
    continue
  fi

  lora_name="${arm_name}-posthoc-step-${step}"

  curl --silent --show-error --fail \
    --request POST \
    --header "Authorization: Bearer $VLLM_API_KEY" \
    --header 'Content-Type: application/json' \
    --data "{\"lora_name\":\"$lora_name\",\"lora_path\":\"$adapter_dir\"}" \
    "http://127.0.0.1:$port/v1/load_lora_adapter"
  echo
  run_eval "step_$step" "$lora_name" "$step" "$adapter_dir"
  curl --silent --show-error --fail \
    --request POST \
    --header "Authorization: Bearer $VLLM_API_KEY" \
    --header 'Content-Type: application/json' \
    --data "{\"lora_name\":\"$lora_name\"}" \
    "http://127.0.0.1:$port/v1/unload_lora_adapter"
  echo
  evaluated=$((evaluated + 1))
done

if [[ $evaluated -eq 0 ]]; then
  echo "No stable 10-step snapshots were available for $arm_name"
fi
