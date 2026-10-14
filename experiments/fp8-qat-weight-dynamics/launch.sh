#!/usr/bin/env bash
# Usage: [WEIGHT_DYNAMICS=0] launch.sh <smoke|pilot|main> <offgrid|ongrid-ue8m0> <bf16|qat-ue8m0|qat-amax> <steps> [extra sft CLI args...]
set -euo pipefail
stage=$1 start=$2 mode=$3 steps=$4
shift 4
here=$(cd "$(dirname "$0")" && pwd)
name="q30b-base_${start}_train-${mode}_muon-1e-5_${steps}st_8k-x16"
[[ $stage != main ]] && name="${stage}-${name}"
weight_dynamics=${WEIGHT_DYNAMICS:-1}
[[ $weight_dynamics == 0 ]] && name="${name}_no-wd-tracker"
warmup=$(( (steps + 8) / 16 ))
(( warmup < 1 )) && warmup=1
overlays=(@ "$here/configs/base.toml" @ "$here/configs/start-$start.toml" @ "$here/configs/train-$mode.toml")
[[ $stage != smoke ]] && overlays+=(@ "$here/configs/ckpt.toml")
tags="[\"stage:$stage\",\"model:q30b-base\",\"start:$start\",\"train:$mode\",\"optim:muon-1e-5\",\"steps:$steps\",\"data:8k-x16\"]"
cd "$here/../.."
exec uv run --no-sync sft "${overlays[@]}" \
  --max-steps "$steps" --scheduler.warmup-steps "$warmup" \
  --run.name "$name" --monitors.wandb.tags "$tags" \
  --env-vars.PRIME_RL_WEIGHT_DYNAMICS "$weight_dynamics" "$@"
