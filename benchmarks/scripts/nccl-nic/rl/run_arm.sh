#!/bin/bash
# Usage: JOB_ID=<slurm job> TRAIN_HOST=<node> INFER_HOST=<node> OUTPUT_DIR=<dir> run_arm.sh <trainer checkout> <inference checkout> <run name>
# Resolves the RL config with the trainer checkout's rl --dry-run, checks both nodes of the existing allocation are idle,
# runs node.sh on the trainer node (trainer + orchestrator) and the inference node (router + 4 vLLM servers + env worker
# pool), then checks that no process of the run survives. Each checkout needs a synced .venv.
set -uo pipefail

train_arm_dir=$(readlink -f "$1")
infer_arm_dir=$(readlink -f "$2")
run_name=$3

job_id=${JOB_ID:?set JOB_ID to the SLURM job holding both nodes}
train_host=${TRAIN_HOST:?set TRAIN_HOST to the trainer node}
infer_host=${INFER_HOST:?set INFER_HOST to the inference node}
output_dir=${OUTPUT_DIR:?set OUTPUT_DIR to the directory that holds the runs}
harness_dir=$(dirname "$(readlink -f "$0")")
config=$harness_dir/qwen3_30b_a3b_reverse_text.toml
run_dir=$output_dir/$run_name
config_dir=$run_dir/configs/attempt_1/resolved
max_steps=20

node_check() {
    srun --jobid=$job_id --overlap -N1 -n1 -w "$1" bash -s <<'CHECK_SH'
echo "== $(hostname)"
echo "gpu_mem_used_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd,)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
pgrep -u "$(id -u)" -af "[P]RL::|[p]rime_rl|[v]llm|[t]orchrun|[o]rchestrator @|[i]nference @|[e]nv-worker-pool|VLLM|EngineCore" || true
ss -ltnH "( sport = :8000 or sport = :8100 or sport = :8101 or sport = :8102 or sport = :8103 or sport = :29500 or sport = :29501 or sport = :5555 or sport = :29000 )" 2>/dev/null
CHECK_SH
}

busy=0
for host in $train_host $infer_host; do
    report=$(node_check "$host")
    echo "$report"
    max_mem=$(echo "$report" | sed -n 's/^gpu_mem_used_mib //p' | tr , '\n' | sort -n | tail -1)
    lines=$(echo "$report" | grep -v -e "^== " -e "^gpu_mem_used_mib " | grep -c .)
    if (( max_mem > 2000 )) || (( lines > 0 )); then
        busy=1
    fi
done
if (( busy )); then
    echo "preflight failed: a node has GPU memory in use, a leftover process or a bound port" >&2
    exit 2
fi

srun --jobid=$job_id --overlap -N1 -n1 -w "$train_host" bash -c \
    "cd $train_arm_dir && PYTHONPATH=$train_arm_dir/deps/verifiers/environments/reverse_text .venv/bin/rl @ $config --max-steps $max_steps --output-dir $output_dir --run.name $run_name --dry-run" \
    || exit 1
for _ in $(seq 30); do
    [ -f "$config_dir/trainer.json" ] && break
    sleep 2
done
[ -f "$config_dir/trainer.json" ] || exit 1

mkdir -p "$run_dir/bench"
{
    echo "trainer_checkout $train_arm_dir trainer_commit $(git -C "$train_arm_dir" rev-parse --short=9 HEAD)"
    echo "inference_checkout $infer_arm_dir inference_commit $(git -C "$infer_arm_dir" rev-parse --short=9 HEAD)"
    echo "launch_epoch $EPOCHREALTIME"
} > "$run_dir/bench/run_info.txt"

TRAIN_ARM_DIR=$train_arm_dir \
INFER_ARM_DIR=$infer_arm_dir \
RUN_DIR=$run_dir \
CONFIG_DIR=$config_dir \
TRAIN_HOST=$train_host \
INFER_HOST=$infer_host \
MASTER_PORT=29500 \
MAX_STEPS=$max_steps \
RUN_TIMEOUT_S=3000 \
TEARDOWN_TIMEOUT_S=300 \
timeout -k 30 4500 srun --jobid=$job_id --overlap -N2 --ntasks-per-node=1 -w "$train_host,$infer_host" \
    --kill-on-bad-exit=0 --export=ALL "$harness_dir/node.sh"
rc=$?
echo "srun_exit_epoch $EPOCHREALTIME rc $rc" >> "$run_dir/bench/run_info.txt"

sleep 5
run_procs() {
    srun --jobid=$job_id --overlap -N1 -n1 -w "$1" bash -s "$run_name" "$2" <<'RUN_PROCS_SH'
for proc in /proc/[0-9]*; do
    if { tr '\0' '\n' < "$proc/environ"; } 2>/dev/null | grep -qx "PRL_RUN_NAME=$1"; then
        pid=${proc#/proc/}
        echo "leftover $(hostname) pid $pid comm $(cat "$proc/comm" 2>/dev/null)"
        [ "$2" = kill ] && kill -KILL "$pid" 2>/dev/null
    fi
done
true
RUN_PROCS_SH
}
leftover=0
for host in $train_host $infer_host; do
    node_check "$host" | tee -a "$run_dir/bench/postflight.txt"
    report=$(run_procs "$host" list)
    if [ -n "$report" ]; then
        echo "$report" | tee -a "$run_dir/bench/postflight.txt"
        leftover=1
    fi
done
if (( leftover )); then
    echo "postflight: killing leftover processes of $run_name" | tee -a "$run_dir/bench/postflight.txt"
    for host in $train_host $infer_host; do
        run_procs "$host" kill
    done
    sleep 5
    for host in $train_host $infer_host; do
        node_check "$host" | tee -a "$run_dir/bench/postflight.txt"
        run_procs "$host" list | tee -a "$run_dir/bench/postflight.txt"
    done
fi
exit "$rc"
