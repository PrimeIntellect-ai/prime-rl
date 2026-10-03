#!/bin/bash

#SBATCH --job-name=glm53-sft
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=all
#SBATCH --exclusive
#SBATCH --output=/output/golden/launcher/logs/job_%j.log
#SBATCH --error=/output/golden/launcher/logs/job_%j.log

# Exit immediately on any failed command so a crash on any node releases the
# allocation instead of leaving zombie processes holding nodes.
set -e

# Configs
export NUM_TRAIN_NODES=8
export NUM_INFER_NODES=0
export GPUS_PER_NODE=8


# Paths
export PROJECT_DIR=/project
export CONFIG_DIR=/run/configs
export TRAIN_CONFIG_PATH=/run/configs/sft.json
export OUTPUT_DIR=/output/golden
export LAUNCHER_DIR=/output/golden/launcher
export LOG_DIR=/run/logs
export PRL_ATTEMPT_CONFIG_DIR=$CONFIG_DIR
export PRL_ATTEMPT_LOG_DIR=$LOG_DIR
mkdir -p $LOG_DIR/trainer
ln -sfn trainer/node_0.log $LOG_DIR/trainer.log

# Clean exits do not trigger srun --kill-on-bad-exit. Each terminal role
# publishes its successful exit so every node uses the same completion condition.
export TRAINER_DONE=$LAUNCHER_DIR/.sft_trainer_done_$SLURM_JOB_ID

mkdir -p "$LAUNCHER_DIR"
rm -f "$TRAINER_DONE"

export PRL_RUN_ID=run-id
export PRL_RUN_NAME=golden


# Networking. Inference nodes occupy the first ranks, followed by trainer nodes.
HOSTNAMES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
TRAIN_HOSTS=( ${HOSTNAMES[@]:$NUM_INFER_NODES:$NUM_TRAIN_NODES} )
export ALL_HOSTNAMES_STR="${HOSTNAMES[*]}"
export MASTER_ADDR="${TRAIN_HOSTS[0]}"
export MASTER_PORT=29500
echo "HOSTNAMES=${HOSTNAMES[*]}"
echo "TRAIN_HOSTS=${TRAIN_HOSTS[*]}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"



# Install environment
cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

# Shared venv: one sync on the batch node populates it for all nodes.
uv sync --all-extras --all-packages




# Cleanup stale node-local state from prior jobs. Orphan python, torchrun, and
# vLLM processes can otherwise retain GPUs or IPC files after SLURM termination.
srun bash -s <<'CLEANUP_SH'
    pkill -9 -f "[p]ython.*prime_rl" 2>/dev/null
    pkill -9 -f "[t]orchrun" 2>/dev/null
    pkill -9 -f "[v]llm-router" 2>/dev/null
    pkill -9 -f "[v]llm" 2>/dev/null
    pkill -9 -f "[p]rime_rl" 2>/dev/null
    pkill -9 "vllm" 2>/dev/null
    pkill -9 "vllm::.*" 2>/dev/null
    sleep 2
    rm -rf /dev/shm/vllm-* /dev/shm/vllm_* /tmp/vllm-* /tmp/vllm_* /tmp/torch-* /tmp/torchelastic_* 2>/dev/null
    procs=$(ps -eo comm,args | grep -E "python|torchrun|vllm|vllm::" | grep -v grep | wc -l)
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s}")
    echo "[node-cleanup] $(hostname) procs=$procs gpu_mem=${gpu}MiB"
CLEANUP_SH

# Run trainer and online-eval roles in one SLURM step. A non-zero exit on any
# node terminates the other roles. Clean trainer exits let the final eval drain.
srun --kill-on-bad-exit=1 bash -s <<'LAUNCH_SH'
    set -o pipefail

    cd $PROJECT_DIR
    [ -f .env ] && source .env
    source $PROJECT_DIR/.venv/bin/activate
    ulimit -n 65536

    IB_HCA=$(ibv_devinfo 2>/dev/null | sed -n -e "/hca_id/p" -e "/link_layer:/p" | grep -B1 InfiniBand | grep hca_id | sed -e "s/^hca_id://g" | tr -d "[[:blank:]]" | paste -sd,)
    [ -n "$IB_HCA" ] && export NCCL_IB_HCA=$IB_HCA


    TRAIN_NODE_RANK=$SLURM_PROCID

    # Trainer env vars: launcher defaults + [env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export HF_HOME="/home/huggingface"
    export TRITON_CACHE_DIR="/tmp/.cache/triton_cache"

    echo $ALL_HOSTNAMES_STR | tee $LOG_DIR/trainer/node_${TRAIN_NODE_RANK}.log
    (
        WANDB_SHARED_LABEL=trainer uv run torchrun \
            --role=trainer \
            --nnodes=$NUM_TRAIN_NODES \
            --nproc-per-node=$GPUS_PER_NODE \
            --node-rank=$TRAIN_NODE_RANK \
            --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv-id=job_$SLURM_JOB_ID \
            --log-dir=$LOG_DIR/trainer/torchrun \
            --tee=3 \
            --redirects=3 \
            --local-ranks-filter=0 \
            -m prime_rl.trainer.sft.train \
            @ $TRAIN_CONFIG_PATH \
            2>&1 | sed -u 's/^\[[a-zA-Z]*[0-9]*\]://' | tee -a $LOG_DIR/trainer/node_${TRAIN_NODE_RANK}.log
        COMPONENT_CODE=$?
        if [ "$COMPONENT_CODE" -eq 0 ] && [ "$TRAIN_NODE_RANK" -eq 0 ]; then
            touch "$TRAINER_DONE" || exit $?
        fi
        exit "$COMPONENT_CODE"
    ) &


# Every node stays in the shared srun step until all terminal roles finish.
( until [ -f "$TRAINER_DONE" ]; do sleep 10; done ) &
COMPLETION_PID=$!

# Ignore clean component exits until the shared completion condition is met.
# Propagate failures through --kill-on-bad-exit=1.
while true; do
    wait -n -p FINISHED_PID
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -ne 0 ] || [ "${FINISHED_PID:-}" = "$COMPLETION_PID" ]; then
        break
    fi
done

REMAINING_PIDS=$(jobs -p)
echo "[$(hostname)] completion condition met with code $EXIT_CODE - terminating remaining processes"
for pid in $REMAINING_PIDS; do
    kill -TERM $pid 2>/dev/null || true
done
exit $EXIT_CODE
LAUNCH_SH
