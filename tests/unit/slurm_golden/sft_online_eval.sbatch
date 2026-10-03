#!/bin/bash

#SBATCH --job-name=prime-rl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/output/golden/launcher/logs/job_%j.log
#SBATCH --error=/output/golden/launcher/logs/job_%j.log

# Exit immediately on any failed command so a crash on any node releases the
# allocation instead of leaving zombie processes holding nodes.
set -e

# Configs
export NUM_TRAIN_NODES=2
export NUM_INFER_NODES=2
export GPUS_PER_NODE=8
export NODES_PER_INFER_REPLICA=1
export NUM_INFER_REPLICAS=2
export ROUTER_PORT=8000
export INFERENCE_TP=1
export INFERENCE_DP_LOCAL=$((GPUS_PER_NODE / INFERENCE_TP))
export INFERENCE_DATA_PARALLEL_RPC_PORT=13345
export BACKEND_PORT=8100
export INFERENCE_ENABLE_EXPERT_PARALLEL=0

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
mkdir -p $LOG_DIR/inference
ln -sfn inference/node_0.log $LOG_DIR/inference.log

# Clean exits do not trigger srun --kill-on-bad-exit. Each terminal role
# publishes its successful exit so every node uses the same completion condition.
export TRAINER_DONE=$LAUNCHER_DIR/.sft_trainer_done_$SLURM_JOB_ID
export EVAL_DONE=$LAUNCHER_DIR/.sft_eval_done_$SLURM_JOB_ID
mkdir -p "$LAUNCHER_DIR"
rm -f "$TRAINER_DONE" "$EVAL_DONE"

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

INFER_HOSTS=( ${HOSTNAMES[@]:0:$NUM_INFER_NODES} )

# Single global router on inference node 0 fronts every per-rank endpoint;
# admin URLs bypass it and hit each rank directly.
INFER_URLS="http://${INFER_HOSTS[0]}:${ROUTER_PORT}/v1"
ADMIN_URLS=""
ROUTER_ARGS=""
# External-LB: one endpoint per DP rank (BACKEND_PORT + dp_local) on each node.
for ((n=0; n<NUM_INFER_NODES; n++)); do
    host="${INFER_HOSTS[$n]}"
    for ((k=0; k<INFERENCE_DP_LOCAL; k++)); do
        ADMIN_URLS="${ADMIN_URLS:+$ADMIN_URLS }http://${host}:$((BACKEND_PORT + k))/v1"
        ROUTER_ARGS="$ROUTER_ARGS http://${host}:$((BACKEND_PORT + k))"
    done
done
export INFER_URLS
export ADMIN_URLS
export ROUTER_ARGS
echo "INFER_HOSTS=${INFER_HOSTS[*]}"
echo "INFER_URLS=${INFER_URLS}"
echo "ADMIN_URLS=${ADMIN_URLS}"
export HOSTNAMES_STR="${INFER_HOSTS[*]}"

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

if [ "$SLURM_PROCID" -lt "$NUM_INFER_NODES" ]; then

    INFER_NODE_RANK=$SLURM_PROCID
    read -ra HOSTNAMES <<< "$HOSTNAMES_STR"
    # Inference env vars: launcher defaults + config env_vars
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
    export VLLM_ENGINE_READY_TIMEOUT_S="4200"
    export UCX_TLS="all"
    LOCAL_IP=$(hostname -I | awk "{print \$1}")

    REPLICA_IDX=$((INFER_NODE_RANK / NODES_PER_INFER_REPLICA))
    RANK_IN_REPLICA=$((INFER_NODE_RANK % NODES_PER_INFER_REPLICA))

    echo "INFER_NODE_RANK=$INFER_NODE_RANK REPLICA=$REPLICA_IDX RANK_IN_REPLICA=$RANK_IN_REPLICA LOCAL_IP=$LOCAL_IP" \
        | tee $LOG_DIR/inference/node_${INFER_NODE_RANK}.log

    CONFIG_PATH="$CONFIG_DIR/inference.json"

# rank_log <node_rank> <k> — DP rank 0 logs to the node-level file (node_<rank>.log,
# the target of the top-level inference.log symlink) so it captures real server output;
# ranks 1+ get their own node_<rank>_rank<k>.log.
rank_log() {
    if [ "$2" -eq 0 ]; then
        echo "$LOG_DIR/inference/node_$1.log"
    else
        echo "$LOG_DIR/inference/node_$1_rank$2.log"
    fi
}

# launch_inference_rank <port> <cuda_visible_devices> <dp> <rpc_port> <vllm_extra_json|""> <nixl_side_channel_port|""> <log_file>
launch_inference_rank() {
    local port="$1" gpus="$2" dp="$3" rpc="$4" extra="$5" nixl="$6" log="$7"
    # Per-rank dir for the mooncake-store lookup ipc socket. It must be unique per process:
    # external-LB runs one engine per process, all with data_parallel_rank_local 0, so the
    # lookup socket name collides on bind if the dir is shared node-wide. $port is unique per
    # rank on the node. Node-local /tmp also avoids inheriting an absent submit-node TMPDIR.
    local rpcbase="/tmp/vllm-rpc-$USER-$SLURM_JOB_ID-$port"
    mkdir -p "$rpcbase"
    local -a cmd=(uv run inference @ "$CONFIG_PATH"
        --server.host 0.0.0.0 --server.port "$port"
        --vllm.data-parallel-size "$dp" --vllm.data-parallel-size-local 1 --vllm.api-server-count 1)
    [ -n "$rpc" ] && cmd+=(--vllm.data-parallel-rpc-port "$rpc")
    # Per-rank vLLM overrides go through a JSON file merged into [inference.vllm]
    # (dict-valued keys like compilation_config can't be inlined as CLI flags).
    if [ -n "$extra" ]; then
        printf '%s' "$extra" > "$rpcbase/vllm-overrides.json"
        cmd+=(--vllm @ "$rpcbase/vllm-overrides.json")
    fi
    if [ -n "$nixl" ]; then
        CUDA_VISIBLE_DEVICES="$gpus" VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl" VLLM_RPC_BASE_PATH="$rpcbase" "${cmd[@]}" 2>&1 | tee -a "$log" &
    else
        CUDA_VISIBLE_DEVICES="$gpus" VLLM_RPC_BASE_PATH="$rpcbase" "${cmd[@]}" 2>&1 | tee -a "$log" &
    fi
}


# launch_router <mode:pd|regular> <worker_args> <port> <policy> <log>
launch_router() {
    local mode="$1" worker_args="$2" port="$3" policy="$4" log="$5"
    local tag="vllm-router"; [ "$mode" = pd ] && tag="vllm-router (PD)"
    echo "Starting $tag on $LOCAL_IP:$port" | tee "$log"
    local -a cmd=(vllm-router --policy "$policy" --host 0.0.0.0 --port "$port"
        --request-id-headers x-session-id
        --request-timeout-secs 14400
        --prometheus-port "$((port + 21000))"
        --worker-startup-timeout-secs 4200 --log-level debug)
    if [ "$mode" = pd ]; then
        cmd+=(--vllm-pd-disaggregation $worker_args)
    else
        cmd+=(--worker-urls $worker_args)
    fi
    "${cmd[@]}" >> "$log" 2>&1 &
}

    # Start the single global router on inference node 0 (external LB across all per-rank endpoints)
    if [ "$INFER_NODE_RANK" -eq 0 ]; then
        launch_router regular "$ROUTER_ARGS" "$ROUTER_PORT" "sticky_least_loaded" "$LOG_DIR/inference/router.log"
    fi

    # External-LB: one vLLM API server per DP rank on this node (BACKEND_PORT + d),
    # each pinned to its TP slice of GPUs; the router balances across the per-rank
    # endpoints. For MoE the ranks form one external-LB DP group (EP all2all spans
    # them) rooted on the replica-head DP coordinator. data_parallel_address is
    # per-node, not per-rank: every rank on the head reaches the coordinator via the
    # head LOCAL_IP, ranks on other nodes via the head hostname (matches the P/D path).
    REPLICA_HEAD_HOST="${HOSTNAMES[$((REPLICA_IDX * NODES_PER_INFER_REPLICA))]}"
    INFER_GROUP_DP=$((NODES_PER_INFER_REPLICA * INFERENCE_DP_LOCAL))
    if [ "$RANK_IN_REPLICA" -eq 0 ]; then DP_ADDR="$LOCAL_IP"; else DP_ADDR="$REPLICA_HEAD_HOST"; fi
    for ((d=0; d<INFERENCE_DP_LOCAL; d++)); do
        RANK_GPUS=$(seq -s, $((d * INFERENCE_TP)) $((d * INFERENCE_TP + INFERENCE_TP - 1)))
        RANK_LOG=$(rank_log "$INFER_NODE_RANK" "$d")
        if [ "$INFERENCE_ENABLE_EXPERT_PARALLEL" -eq 1 ] && [ "$INFER_GROUP_DP" -gt 1 ]; then
            GLOBAL_DP_RANK=$((RANK_IN_REPLICA * INFERENCE_DP_LOCAL + d))
            RANK_EXTRA="{\"data_parallel_rank\": $GLOBAL_DP_RANK, \"data_parallel_address\": \"$DP_ADDR\"}"
            launch_inference_rank "$((BACKEND_PORT + d))" "$RANK_GPUS" "$INFER_GROUP_DP" "$INFERENCE_DATA_PARALLEL_RPC_PORT" "$RANK_EXTRA" "" "$RANK_LOG"
        else
            # Dense or single-DP-rank EP (dp=1, e.g. EP within one tp=8 group):
            # independent single-engine servers. vLLM rejects the external-LB DP
            # args when data_parallel_size == 1; enable_expert_parallel in the
            # inference config alone shards experts across the TP group.
            launch_inference_rank "$((BACKEND_PORT + d))" "$RANK_GPUS" 1 "" "" "" "$RANK_LOG"
        fi
    done

    if [ "$INFER_NODE_RANK" -eq 0 ]; then

        mkdir -p $LOG_DIR/envs/eval
        uv run env-server @ $CONFIG_DIR/envs/eval/reverse-text.json \
            > $LOG_DIR/envs/eval/reverse-text.log 2>&1 &

        # Online evals drive inference through the router and send admin operations
        # directly to each engine.
        export CUDA_DEVICE_ORDER="PCI_BUS_ID"
        export PYTHONUNBUFFERED="1"
        export OMP_NUM_THREADS="1"
        export GIT_LFS_SKIP_SMUDGE="1"
        export LOGURU_FORCE_COLORS="1"
        (
            PRL_LOG_DIR=$LOG_DIR WANDB_SHARED_LABEL=online-eval uv run python -m prime_rl.eval.online \
                @ $CONFIG_DIR/eval.json \
                --client.base-url $INFER_URLS \
                --client.admin-base-url $ADMIN_URLS \
                --weight-broadcast.host $MASTER_ADDR \
                2>&1 | tee $LOG_DIR/eval.log
            COMPONENT_CODE=$?
            if [ "$COMPONENT_CODE" -eq 0 ]; then
                touch "$EVAL_DONE" || exit $?
            fi
            exit "$COMPONENT_CODE"
        ) &
    fi
else
    TRAIN_NODE_RANK=$((SLURM_PROCID - NUM_INFER_NODES))

    # Trainer env vars: launcher defaults + [env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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
            --weight-broadcast.host 0.0.0.0 \
            2>&1 | sed -u 's/^\[[a-zA-Z]*[0-9]*\]://' | tee -a $LOG_DIR/trainer/node_${TRAIN_NODE_RANK}.log
        COMPONENT_CODE=$?
        if [ "$COMPONENT_CODE" -eq 0 ] && [ "$TRAIN_NODE_RANK" -eq 0 ]; then
            touch "$TRAINER_DONE" || exit $?
        fi
        exit "$COMPONENT_CODE"
    ) &
fi

# Every node stays in the shared srun step until all terminal roles finish.
( until [ -f "$TRAINER_DONE" ] && [ -f "$EVAL_DONE" ]; do sleep 10; done ) &
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
