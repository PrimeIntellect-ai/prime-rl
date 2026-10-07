#!/bin/bash
#SBATCH --job-name=score-centering-pair
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=116
#SBATCH --gres=gpu:8
#SBATCH --partition=all
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --output=experiments/score-centering/results/monitor/ipo_pair_%j.log
#SBATCH --error=experiments/score-centering/results/monitor/ipo_pair_%j.log

set -euo pipefail
cd /home/mika/.herdr/worktrees/prime-rl/feat-score-centering
mapfile -t pair_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#pair_hosts[@]} -eq 4 ]]

run_arm() (
    arm=$1
    export PRL_PAIR_NODELIST=$2
    export SLURM_JOB_NODELIST=$PRL_PAIR_NODELIST
    export SLURM_NODELIST=$PRL_PAIR_NODELIST
    export SLURM_JOB_NUM_NODES=2
    export SLURM_NNODES=2
    export SLURM_NTASKS=2
    export SLURM_TASKS_PER_NODE='1(x2)'
    export SLURM_NTASKS_PER_NODE=1
    unset PRL_RUN_ID WANDB_RUN_ID

    # Generated launchers use srun for node cleanup and component startup.
    srun() {
        command srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
            --nodelist="$PRL_PAIR_NODELIST" --cpus-per-task=116 --exclusive "$@"
    }
    export -f srun
    srun uv run --no-sync python -c 'import os,socket; n=len(os.sched_getaffinity(0)); print(socket.gethostname(), "available CPUs", n, flush=True); assert n >= 116, n'
    run_dir="outputs/score-centering/${arm}-eps03-seed42-compact-v1"
    printf '%s nodes=%s run=%s\n' "$arm" "$PRL_PAIR_NODELIST" "$run_dir"
    bash "$run_dir/launcher/rl.sbatch" > "$run_dir/launcher/logs/job_${SLURM_JOB_ID}.log" 2>&1
)

run_arm ipo "${pair_hosts[0]},${pair_hosts[1]}" &
ipo_pid=$!
run_arm ipo-sc "${pair_hosts[2]},${pair_hosts[3]}" &
sc_pid=$!

# Keep the allocation alive until both arms exit, even if one arm fails.
set +e
wait "$ipo_pid"
ipo_status=$?
wait "$sc_pid"
sc_status=$?
set -e

printf 'ipo exit=%s, ipo-sc exit=%s\n' "$ipo_status" "$sc_status"
(( ipo_status == 0 && sc_status == 0 ))
