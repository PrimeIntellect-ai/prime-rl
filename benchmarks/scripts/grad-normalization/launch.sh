#!/usr/bin/env bash
# Launch one gradient-normalization A/B arm with the probes and the emulated MX encoder installed.
#
# Usage: launch.sh --worktree PATH --gpus IDS --layout single|cp2|dp2|offload --out DIR --name NAME
#                  [--normalize grads|loss] [--emin N] [--encoder 0|1] [--encoder-sites dy,dz,sdy,sdz,ao,aqkv]
#                  [--emin-scan N,N,...] [--steps N] [--grad-dump-step N] [--fake] [-- EXTRA_ARGS...]
#
# Runs `uv run --no-sync rl` (or the trainer alone on seeded fake data with --fake) from the arm's worktree.
# The first GPU in --gpus serves inference, the rest train; it also picks the server and rollout ports, so arms
# on disjoint GPUs can run side by side. Omit --normalize for a commit without the flag.
# Outputs go to OUT/NAME; probe records, gradient dumps, commit and command to OUT/NAME.gn/.
set -euo pipefail

harness_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
worktree="" gpus="" layout="single" out="" name="" normalize="" emin="-127" encoder="0" encoder_sites="dy,dz"
emin_scan="" steps="" grad_dump_step="" fake=0
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --worktree) worktree="$2"; shift 2 ;;
        --gpus) gpus="$2"; shift 2 ;;
        --layout) layout="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        --name) name="$2"; shift 2 ;;
        --normalize) normalize="$2"; shift 2 ;;
        --emin) emin="$2"; shift 2 ;;
        --encoder) encoder="$2"; shift 2 ;;
        --encoder-sites) encoder_sites="$2"; shift 2 ;;
        --emin-scan) emin_scan="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --grad-dump-step) grad_dump_step="$2"; shift 2 ;;
        --fake) fake=1; shift ;;
        --) shift; extra=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for required in worktree gpus out name; do
    if [[ -z "${!required}" ]]; then
        echo "--${required//_/-} is required" >&2
        exit 2
    fi
done

if [[ $fake == 1 ]]; then
    prefix=""
else
    prefix="trainer."
fi
case "$layout" in
    single) train_gpus=1; layout_args=() ;;
    cp2) train_gpus=2; layout_args=(--${prefix}model.cp 2 --${prefix}model.ep 1) ;;
    dp2) train_gpus=2; layout_args=(--${prefix}model.ep 1) ;;
    offload) train_gpus=1; layout_args=(--${prefix}model.offload full) ;;
    *) echo "unknown layout: $layout" >&2; exit 2 ;;
esac

IFS=, read -r -a gpu_ids <<< "$gpus"
needed=$((train_gpus + 1 - fake))
if [[ ${#gpu_ids[@]} -ne $needed ]]; then
    echo "layout $layout needs $needed GPUs, got ${#gpu_ids[@]}: $gpus" >&2
    exit 2
fi

run_dir="$out/$name"
server_port=$((9000 + 10 * gpu_ids[0]))
probe_dir="$run_dir.gn"
mkdir -p "$probe_dir"

gn_env=(
    "PYTHONPATH=$harness_dir/inject"
    "GN_OUT=$probe_dir"
    "GN_EMIN=$emin"
    "GN_ENCODER=$encoder"
    "GN_ENCODER_SITES=$encoder_sites"
    "GN_EMIN_SCAN=$emin_scan"
    "GN_GRAD_DUMP_STEP=$grad_dump_step"
)

args=("${layout_args[@]}")
[[ -n "$normalize" ]] && args+=(--${prefix}optim.normalize "$normalize")
[[ -n "$steps" ]] && args+=(--max-steps "$steps")

cd "$worktree"
git rev-parse HEAD > "$probe_dir/commit.txt"
export CUDA_VISIBLE_DEVICES="$gpus"

if [[ $fake == 1 ]]; then
    export "${gn_env[@]}"
    cmd=(uv run --no-sync torchrun --standalone --nproc-per-node "$train_gpus" -m prime_rl.trainer.rl.train
        @ "$harness_dir/trainer_fake.toml" --output-dir "$run_dir" --no-monitors.prime "${args[@]}" "${extra[@]}")
else
    trainer_env="{$(for pair in "${gn_env[@]}"; do printf '"%s": "%s",' "${pair%%=*}" "${pair#*=}"; done)"
    trainer_env="${trainer_env%,}}"
    cmd=(uv run --no-sync rl @ "$harness_dir/rl.toml" --output-dir "$out" --run.name "$name"
        --deployment.num-train-gpus "$train_gpus" --deployment.num-infer-gpus 1
        --inference.server.port "$server_port" --orchestrator.model.client.base-url "http://localhost:$server_port/v1"
        --rollout-transport.type zmq
        --rollout-transport.port $((5600 + gpu_ids[0]))
        --trainer.env-vars "$trainer_env" --no-monitors.prime "${args[@]}" "${extra[@]}")
fi
printf '%q ' "${cmd[@]}" > "$probe_dir/launch_command.txt"
echo >> "$probe_dir/launch_command.txt"
exec "${cmd[@]}"
