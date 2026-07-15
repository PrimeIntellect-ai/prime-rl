#!/usr/bin/env bash

set -euo pipefail

mode=${1:-smoke}
root=/home/ubuntu/prime-rl
manifest=$root/configs/opd-gap/qualification/math-family-qwen3-r01-manifest.json
state_dir=$root/evals/math-family-qwen3-r01-20260715
mkdir -p "$state_dir"

export PRIME_API_KEY
export PRIME_TEAM_ID
PRIME_API_KEY=$(jq -r .api_key /home/ubuntu/.prime/config.json)
PRIME_TEAM_ID=$(jq -r .team_id /home/ubuntu/.prime/config.json)

models=(
  Qwen/Qwen3-0.6B
  Qwen/Qwen3-1.7B
  Qwen/Qwen3-4B
  Qwen/Qwen3-8B
  Qwen/Qwen3-14B
  Qwen/Qwen3-32B
)

launch() {
  local model=$1 band=$2 lo=$3 hi=$4 examples=$5 rollouts=$6
  local hosted_model=$model
  if [[ $model == Qwen/Qwen3-8B ]]; then
    hosted_model=qwen/qwen3-8b
  fi
  local safe=${model#Qwen/Qwen3-}
  safe=${safe,,}
  local name="opd-gap-math-${safe}-${band}-k${rollouts}-r01"
  local args
  args=$(jq -cn \
    --argjson lo "$lo" --argjson hi "$hi" \
    '{dataset_name:"PrimeIntellect/INTELLECT-3-RL",dataset_subset:"math",dataset_split:"train",dataset_shuffle:true,dataset_seed:20260715,difficulty_key:"avg@8_qwen3_4b_thinking_2507",min_avg_reward:$lo,max_avg_reward:$hi,judge_model:null,math_verify_timeout:60}')
  local output
  output=$(prime eval run primeintellect/math-env@0.1.6 \
    --hosted --eval-name "$name" --model "$hosted_model" \
    --env-args "$args" --num-examples "$examples" --rollouts-per-example "$rollouts" \
    --max-concurrent "$rollouts" --max-tokens 4096 --temperature 1.0 \
    --max-retries 2 2>&1) || true
  printf '%s\n' "$output" | tee "$state_dir/${name}.launch.log"
  local eval_id
  eval_id=$(sed -nE 's/.*\b([a-z0-9]{24})\b.*/\1/p' <<<"$output" | tail -1)
  jq -cn --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg mode "$mode" --arg model "$model" --arg hosted_model "$hosted_model" --arg band "$band" \
    --arg name "$name" --arg eval_id "$eval_id" --arg output "$output" \
    '{timestamp_utc:$timestamp,mode:$mode,model:$model,hosted_model:$hosted_model,band:$band,eval_name:$name,eval_id:$eval_id,launch_output:$output}' \
    >>"$state_dir/launches.jsonl"
  [[ -n "$eval_id" ]]
}

case "$mode" in
  smoke)
    for model in "${models[@]}"; do
      launch "$model" mid 0.375 0.625 1 2 || true
    done
    ;;
  smoke-8b)
    launch Qwen/Qwen3-8B mid 0.375 0.625 1 2
    ;;
  full)
    for model in "${models[@]}"; do
      launch "$model" zero 0.0 0.0 16 16
      launch "$model" low 0.125 0.25 16 16
      launch "$model" mid 0.375 0.625 16 16
      launch "$model" high 0.75 1.0 16 16
    done
    ;;
  *)
    echo "usage: $0 {smoke|smoke-8b|full}" >&2
    exit 2
    ;;
esac
