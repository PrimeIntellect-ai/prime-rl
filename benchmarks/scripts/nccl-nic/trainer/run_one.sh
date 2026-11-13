#!/bin/bash
# Usage: JOB_ID=<slurm job> NODES=<node0,node1> MAIN_CHECKOUT=<dir> FIX_CHECKOUT=<dir> OUTPUT_DIR=<dir> \
#     run_one.sh <arm: main|fix> <run name> <master port> [config toml, default qwen3_30b_a3b_fake.toml]
# Resolves the config with the arm's own sft --dry-run, then runs node.sh on both nodes of the job.
set -uo pipefail

arm=$1
run_name=$2
master_port=$3
config=${4:-qwen3_30b_a3b_fake.toml}

job_id=$JOB_ID
IFS=, read -ra nodes <<< "$NODES"
harness_dir=$(dirname "$(readlink -f "$0")")
case $arm in
    main) arm_dir=$MAIN_CHECKOUT ;;
    fix) arm_dir=$FIX_CHECKOUT ;;
    *) echo "arm must be main or fix" >&2; exit 1 ;;
esac
output_dir=$OUTPUT_DIR
run_dir=$output_dir/$run_name
max_steps=20

srun --jobid=$job_id --overlap -N1 -n1 -w "${nodes[0]}" bash -c \
    "cd $arm_dir && .venv/bin/sft @ $harness_dir/$config --max-steps $max_steps --output-dir $output_dir --run.name $run_name --dry-run" \
    || exit 1
srun --jobid=$job_id --overlap -N1 -n1 -w "${nodes[0]}" test -f "$run_dir/configs/attempt_1/resolved/trainer.json" || exit 1

mkdir -p "$run_dir/bench"
echo "arm $arm commit $(git -C "$arm_dir" rev-parse --short=9 HEAD) launch_epoch $EPOCHREALTIME" > "$run_dir/bench/run_info.txt"

ARM_DIR=$arm_dir \
RUN_DIR=$run_dir \
TRAIN_CONFIG_PATH=$run_dir/configs/attempt_1/resolved/trainer.json \
MASTER_ADDR=${nodes[0]} \
MASTER_PORT=$master_port \
MAX_STEPS=$max_steps \
RUN_TIMEOUT_S=2700 \
TEARDOWN_TIMEOUT_S=300 \
timeout 3000 srun --jobid=$job_id --overlap -N2 --ntasks-per-node=1 -w "$(IFS=,; echo "${nodes[*]}")" \
    --kill-on-bad-exit=0 --export=ALL "$harness_dir/node.sh"
rc=$?
echo "srun_exit_epoch $EPOCHREALTIME rc $rc" >> "$run_dir/bench/run_info.txt"
exit "$rc"
