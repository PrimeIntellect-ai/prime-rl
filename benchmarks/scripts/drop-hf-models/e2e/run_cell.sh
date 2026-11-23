#!/usr/bin/env bash
set -uo pipefail

HARNESS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HARNESS/env.sh"
source "$HARNESS/lib.sh"

usage() {
  echo "usage: run_cell.sh <main1|main2|pr1|pr2> <cell> [--dry-run]" >&2
  echo "cells: ${ALL_CELLS[*]}" >&2
  exit 2
}

[ $# -ge 2 ] || usage
arm=$1
cell=$2
dry_run=${3:-}
worktree=$(arm_worktree "$arm") || usage
if [ -f "$E2E_OUT/$arm/skip-$cell" ]; then
  echo "skipping $arm/$cell (skip-$cell present)"
  exit 0
fi
setup_uv "$worktree"

kind=sft
configs=()
extra=()
case $cell in
  sft-dense) configs=(sft-dense.toml) ;;
  sft-glm) configs=(sft-glm.toml) ;;
  sft-glm-cp2) configs=(sft-glm.toml cp2.toml) ;;
  sft-qwen-moe) configs=(sft-qwen-moe.toml) ;;
  sft-vlm)
    configs=(sft-vlm.toml)
    extra=(--data.source "[{\"name\": \"vlm-synth\", \"dataset\": \"$E2E_DATA/vlm-synth\"}]")
    ;;
  ckpt-dense-write) configs=(sft-dense.toml ckpt.toml) ;;
  ckpt-moe-write) configs=(sft-moe-fake.toml ckpt.toml) ;;
  ckpt-dense-resume)
    configs=(sft-dense.toml ckpt.toml)
    extra=(--resume.dir "$E2E_OUT/main1/ckpt-dense-write/checkpoints/step_10")
    ;;
  ckpt-moe-resume)
    configs=(sft-moe-fake.toml ckpt.toml)
    extra=(--resume.dir "$E2E_OUT/main1/ckpt-moe-write/checkpoints/step_10")
    ;;
  rl-dense-nccl) kind=rl; configs=(rl-dense.toml) ;;
  rl-dense-fs) kind=rl; configs=(rl-dense.toml fs.toml) ;;
  rl-moe-nccl | rl-moe-fs)
    kind=rl
    configs=(rl-moe.toml)
    [ "$cell" = rl-moe-fs ] && configs+=(fs.toml)
    conversion_dir=$E2E_OUT/$arm/conv/$cell
    extra=(--trainer.model.conversion-dir "$conversion_dir")
    ;;
  *) usage ;;
esac

config_args=()
for config in "${configs[@]}"; do
  config_args+=(@ "$HARNESS/configs/$config")
done

if [ "$dry_run" = --dry-run ]; then
  out=$E2E_OUT/dry/$arm
  mkdir -p "$out"
  cd "$worktree" && $UV_RUN "$kind" "${config_args[@]}" "${extra[@]}" \
    --output-dir "$out" --run.name "$cell" --clean --no-monitors.prime --no-dashboard --dry-run
  exit $?
fi

out=$E2E_OUT/$arm
run_dir=$out/$cell
inductor_dir=$E2E_SCRATCH/inductor/$arm-$cell
mkdir -p "$out"
rm -rf "$run_dir" "$out/$cell.probe" "$inductor_dir"
[ -n "${conversion_dir:-}" ] && rm -rf "$conversion_dir"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
trainer_torch_logs="recompiles,graph_breaks,dynamo"

if [ "$kind" = rl ] && curl -s -o /dev/null --max-time 3 http://localhost:8000/health; then
  echo "an inference server already answers on :8000; refusing to launch $cell" >&2
  exit 3
fi

sha=$(git -C "$worktree" rev-parse HEAD)
start=$(date +%s)
echo "[$(date -u +%FT%TZ)] $arm/$cell at $sha"

if [ "$kind" = sft ]; then
  (
    cd "$worktree" &&
      TORCH_LOGS=$trainer_torch_logs TORCHINDUCTOR_CACHE_DIR=$inductor_dir \
        $UV_RUN sft "${config_args[@]}" "${extra[@]}" \
        --output-dir "$out" --run.name "$cell" --clean --no-monitors.prime --no-dashboard
  ) >"$out/$cell.launch.log" 2>&1
  rc=$?
else
  PRIME_RL_ADMIN_TOKEN=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
  export PRIME_RL_ADMIN_TOKEN
  probe_args=("$run_dir" "$out/$cell.probe")
  [[ $cell == *-fs ]] && probe_args+=(--filesystem)
  "$PY" "$HARNESS/weight_probe.py" "${probe_args[@]}" >"$out/$cell.probe.log" 2>&1 &
  probe_pid=$!
  trainer_env="{\"TORCH_LOGS\": \"$trainer_torch_logs\", \"TORCHINDUCTOR_CACHE_DIR\": \"$inductor_dir\"}"
  (
    cd "$worktree" &&
      $UV_RUN rl "${config_args[@]}" "${extra[@]}" --trainer.env-vars "$trainer_env" \
        --output-dir "$out" --run.name "$cell" --clean --no-monitors.prime --no-dashboard
  ) >"$out/$cell.launch.log" 2>&1
  rc=$?
  kill "$probe_pid" 2>/dev/null
  wait "$probe_pid"
fi

end=$(date +%s)
printf '%s\t%s\t%s\t%s\t%s\n' "$cell" "$rc" "$sha" "$(date -u -d "@$start" +%FT%TZ)" "$((end - start))" >>"$out/status.tsv"
echo "[$(date -u +%FT%TZ)] $arm/$cell rc=$rc in $((end - start))s"
exit "$rc"
