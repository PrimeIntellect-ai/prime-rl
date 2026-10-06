#!/usr/bin/env bash
# Submit every ablation run not yet present; each fp32-cap2-r2 run avoids the node pair its fp32-cap2 run used.
set -euo pipefail

AB_DIR=$(cd "$(dirname "$0")" && pwd)
RUNS_DIR=$HOME/tmp/router-rs-cap/runs
BASE_EXCLUDE=prime-nebius-puku-h200-gpu-001
SCENARIOS=(s4k-ac s4k-noac s8k-noac s1k-noac)
LAYOUTS=(fsdp16 ep8)
VARIANTS=(fp32-cap1 fp32-cap2 fp32-cap3 bf16-cap1 bf16-cap2)

submit() {
  local scenario=$1 layout=$2 variant=$3
  shift 3
  [ -d "$RUNS_DIR/$scenario-$layout-$variant" ] && return 0
  SCENARIO=$scenario "$AB_DIR/launch.sh" "$layout" "$variant" "$@" 2>&1 \
    | grep -oE "Submitted batch job [0-9]+|Error.*" | sed "s/^/$scenario-$layout-$variant: /"
}

for scenario in "${SCENARIOS[@]}"; do
  for layout in "${LAYOUTS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      submit "$scenario" "$layout" "$variant"
    done
  done
done

for scenario in "${SCENARIOS[@]}"; do
  for layout in "${LAYOUTS[@]}"; do
    node_log=$RUNS_DIR/$scenario-$layout-fp32-cap2/logs/attempt_1/trainer/node_0.log
    until [ -s "$node_log" ]; do sleep 20; done
    cap2_nodes=$(head -1 "$node_log" | tr ' ' ',')
    submit "$scenario" "$layout" fp32-cap2-r2 --slurm.exclude "$BASE_EXCLUDE,$cap2_nodes"
  done
done
