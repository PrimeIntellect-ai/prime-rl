#!/bin/bash

#SBATCH --job-name=prime-rl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive
#SBATCH --output=/output/launcher/logs/job_%j.log
#SBATCH --error=/output/launcher/logs/job_%j.log

# Exit immediately on any failed command so a crash anywhere releases the
# allocation instead of leaving zombie processes holding nodes.
set -e

# Configs
export NUM_NODES=2
export GPUS_PER_NODE=8

export ROUTER_PORT=8000
export BACKEND_PORT=8100
export RPC_PORT=13345

# Paths
export PROJECT_DIR=/project
export CONFIG_PATH=/run/configs/inference.json
export CONFIG_DIR=/run/configs
export OUTPUT_DIR=/output
export LOG_DIR=/run/logs
export PRL_ATTEMPT_CONFIG_DIR=$CONFIG_DIR
export PRL_ATTEMPT_LOG_DIR=$LOG_DIR
mkdir -p $LOG_DIR/inference
ln -sfn inference/node_0.log $LOG_DIR/inference.log

# Setup environment
cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

# Shared (NFS) venv: one sync on the batch node populates it for all nodes.
uv sync --all-extras --all-packages


# Print inference URLs
export HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export HOSTNAMES_STR="${HOSTNAMES[*]}"

# External LB: one router on node 0; one endpoint per DP rank (BACKEND_PORT + dp_local) on every node.
INFER_URLS="http://${HOSTNAMES[0]}:${ROUTER_PORT}/v1"
DP_PER_NODE=2
ROUTER_ARGS=""
for host in ${HOSTNAMES[@]}; do
    for ((d=0; d<DP_PER_NODE; d++)); do
        ROUTER_ARGS="$ROUTER_ARGS http://${host}:$((BACKEND_PORT + d))"
    done
done
export INFER_URLS
export ROUTER_ARGS
echo "HOSTNAMES=${HOSTNAMES[@]}"
echo "INFER_URLS=${INFER_URLS}"



# Cleanup stale node-local state from prior jobs. Orphan python/torchrun/vllm
# processes and vLLM/torch IPC files under /dev/shm and /tmp can survive SLURM
# termination and cause the next launch to hang (e.g. decode engines stuck at
# "Waiting for READY message from DP Coordinator"). Harmless on clean nodes.
#
# `bash -s` with a heredoc on stdin keeps the script body out of every bash
# / srun argv — otherwise `pkill -f "[v]llm"` self-matches the cleanup procs
# because the surviving literal `vllm` strings below appear in the cmdline.
srun bash -s <<'CLEANUP_SH'
    pkill -9 -f "[p]ython.*prime_rl" 2>/dev/null
    pkill -9 -f "[t]orchrun" 2>/dev/null
    pkill -9 -f "[v]llm-router" 2>/dev/null
    # llm-d router procs (EPP + Envoy + decode pd-sidecar) left over from a prior run.
    pkill -9 -f "[e]pp.*pool-name" 2>/dev/null
    pkill -9 -f "[e]nvoy.*envoy.yaml" 2>/dev/null
    pkill -9 -f "[p]d-sidecar" 2>/dev/null
    pkill -9 -f "[v]llm" 2>/dev/null
    pkill -9 -f "[p]rime_rl" 2>/dev/null
    # Also match prctl-set process names (kernel comm) like "vllm::router"
    # which pkill -f does not see because -f only matches the cmdline.
    pkill -9 "vllm" 2>/dev/null
    pkill -9 "vllm::.*" 2>/dev/null
    # Mooncake store (master + client) left over from a prior offload run. -x matches
    # the exact process name, so it never hits the worktree path (e.g. mooncake-baby).
    pkill -9 -x mooncake_master 2>/dev/null
    pkill -9 -x mooncake_client 2>/dev/null
    sleep 2
    rm -rf /dev/shm/vllm-* /dev/shm/vllm_* /tmp/vllm-* /tmp/vllm_* /tmp/torch-* /tmp/torchelastic_* 2>/dev/null
    procs=$(ps -eo comm,args | grep -E "python|torchrun|vllm|vllm::|mooncake_|[e]pp |envoy|pd-sidecar" | grep -v grep | wc -l)
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {print s}")
    echo "[node-cleanup] $(hostname) procs=$procs gpu_mem=${gpu}MiB"
CLEANUP_SH

# Run inference
# --kill-on-bad-exit=1 propagates a non-zero exit on any node to all other
# tasks so a crashed router/engine terminates the SLURM job instead of leaving
# the surviving ranks holding nodes.
#
# The body goes to bash on stdin through a quoted heredoc instead of a
# single-quoted `bash -c` argument: srun broadcasts stdin to every task, and the
# quoted delimiter passes the script through verbatim, so apostrophes and quotes
# (including in the shared _launch_*.sh.j2 includes) need no escaping. Nothing in
# the body reads stdin, which would otherwise eat the rest of the script.
srun --kill-on-bad-exit=1 bash -s <<'LAUNCH_SH'
    # pipefail ensures `cmd | tee log` propagates `cmd` failures, and the
    # wait/cleanup block at the end of this script surfaces background
    # process crashes (router) as task failures.
    set -o pipefail

    cd $PROJECT_DIR
    [ -f .env ] && source .env
    source .venv/bin/activate

    # Infiniband setup
    IB_HCA=$(ibv_devinfo 2>/dev/null | sed -n -e "/hca_id/p" -e "/link_layer:/p" | grep -B1 InfiniBand | grep hca_id | sed -e "s/^hca_id://g" | tr -d "[[:blank:]]" | paste -sd,)
    [ -n "$IB_HCA" ] && export NCCL_IB_HCA=$IB_HCA

    INFER_NODE_RANK=$SLURM_PROCID
    LOCAL_IP=$(hostname -I | awk "{print \$1}")
    # Inference env vars: launcher defaults + [inference.env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
    export VLLM_ENGINE_READY_TIMEOUT_S="4200"
    export UCX_TLS="all"


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



    echo "INFER_NODE_RANK=${INFER_NODE_RANK}, LOCAL_IP=${LOCAL_IP}" \
        | tee $LOG_DIR/inference/node_${INFER_NODE_RANK}.log

    DP_PER_NODE=2
    TP=$((GPUS_PER_NODE / DP_PER_NODE))
    EP=1

    # Start the single global router on node 0 (balances across per-rank endpoints — no intra-node DP header)
    if [ "$INFER_NODE_RANK" -eq 0 ]; then
        launch_router regular "$ROUTER_ARGS" "$ROUTER_PORT" "sticky_least_loaded" "$LOG_DIR/inference/router.log"
    fi

    # External-LB: one vLLM API server per DP rank on this node (BACKEND_PORT + dp_local),
    # each pinned to its TP slice of GPUs. Each node is an independent replica, so any MoE EP
    # group is node-local (rendezvous on LOCAL_IP); dense models run independent engines.
    for ((d=0; d<DP_PER_NODE; d++)); do
        RANK_GPUS=$(seq -s, $((d * TP)) $((d * TP + TP - 1)))
        RANK_LOG=$(rank_log "$INFER_NODE_RANK" "$d")
        if [ "$EP" -eq 1 ] && [ "$DP_PER_NODE" -gt 1 ]; then
            RANK_EXTRA="{\"data_parallel_rank\": $d, \"data_parallel_address\": \"$LOCAL_IP\"}"
            launch_inference_rank "$((BACKEND_PORT + d))" "$RANK_GPUS" "$DP_PER_NODE" "$RPC_PORT" "$RANK_EXTRA" "" "$RANK_LOG"
        else
            # Dense or single-DP-rank EP (dp=1, e.g. EP within one tp=8 group):
            # independent single-engine servers. vLLM rejects the external-LB DP
            # args when data_parallel_size == 1; enable_expert_parallel in the
            # inference config alone shards experts across the TP group.
            launch_inference_rank "$((BACKEND_PORT + d))" "$RANK_GPUS" 1 "" "" "" "$RANK_LOG"
        fi
    done

# Wait for the first background job to exit and propagate its exit code. This
# turns background process crashes (router) into task failures that
# --kill-on-bad-exit=1 can broadcast to the other nodes so the whole job tears
# down instead of leaving zombies holding the allocation.
wait -n
EXIT_CODE=$?
echo "[$(hostname)] component exited with code $EXIT_CODE - terminating remaining processes"
for pid in $(jobs -p); do
    kill -TERM $pid 2>/dev/null || true
done
exit $EXIT_CODE
LAUNCH_SH
