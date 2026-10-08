#!/usr/bin/env bash
# Usage: run_4n.sh <arm-name> <run-name> [extra sft args]. Generates the multi-node sft launcher with
# --dry-run, then runs it as job steps of hold 3409 on the four nodes below (adapted from attach_launch.sh).
set -euo pipefail
NODES=prime-nebius-puku-h200-gpu-009,prime-nebius-puku-h200-gpu-010,prime-nebius-puku-h200-gpu-014,prime-nebius-puku-h200-gpu-016
harness=$(cd "$(dirname "$0")" && pwd)
arm=$HOME/tmp/sft_pad_to_cp/arms/$1; run=$2
runs=$HOME/tmp/sft_pad_to_cp/runs
cd "$arm"
env -u HF_HOME PRL_OUTPUT_DIR="$runs" uv run --no-sync sft @ "$harness/dsv4-sft.toml" --run.name "$run" \
  --monitors.wandb.name "$run" --no-dashboard --dry-run \
  --deployment.type multi_node --deployment.num-train-nodes 4 --slurm.job-name "$run" --slurm.partition all \
  --model.debug.num-layers 21 --data.batch-size 4 "${@:3}"
d=$runs/$run
ORDERED=$(srun --jobid=3409 --overlap -N4 --ntasks-per-node=1 --nodelist=$NODES bash -c 'echo "$SLURM_PROCID $(hostname)"' | sort -n | awk '{print $2}' | paste -sd,)
ATTACH="--jobid=3409 --overlap -N4 --ntasks-per-node=1 --nodelist=$ORDERED"
sed -e "s|^srun bash -s|srun $ATTACH bash -s|" \
    -e "s|^srun --kill-on-bad-exit=1 bash -s|srun $ATTACH --kill-on-bad-exit=1 bash -s|" \
    "$d/launcher/sft.sbatch" > "$d/launcher/sft.attached.sh"
[ "$(grep -c "^srun $ATTACH" "$d/launcher/sft.attached.sh")" = 2 ] || { echo "srun patch count != 2"; exit 1; }
cache=/tmp/garrett/sftpad/$run
export SLURM_JOB_ID=3409 SLURM_JOB_NODELIST="$ORDERED" SLURM_JOB_NUM_NODES=4
export HF_HUB_CACHE=/home/huggingface/hub TRITON_CACHE_DIR=$cache/triton TORCHINDUCTOR_CACHE_DIR=$cache/inductor
export TRITON_PRINT_AUTOTUNING=1 TORCH_LOGS=recompiles
env -u HF_HOME bash "$d/launcher/sft.attached.sh"
