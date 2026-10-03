#!/bin/bash

#SBATCH --job-name=glm5-llmd
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=all
#SBATCH --exclusive
#SBATCH --output=/output/golden/launcher/logs/job_%j.log
#SBATCH --error=/output/golden/launcher/logs/job_%j.log

# Exit immediately on any failed command so a crash anywhere releases the
# allocation instead of leaving zombie processes holding nodes.
set -e

# Configs
export NUM_TRAIN_NODES=16
export NUM_INFER_NODES=2
export NODES_PER_INFER_REPLICA=2
export NUM_INFER_REPLICAS=1
export GPUS_PER_NODE=8
export ROUTER_PORT=8000
export INFERENCE_TP=1
export INFERENCE_DP_LOCAL=$((GPUS_PER_NODE / INFERENCE_TP))
export INFERENCE_DATA_PARALLEL_RPC_PORT=13345
export NUM_PREFILL_NODES=1
export NUM_DECODE_NODES=1
export NUM_PREFILL_REPLICAS=1
export NUM_DECODE_REPLICAS=1
export NODES_PER_PREFILL_REPLICA=1
export NODES_PER_DECODE_REPLICA=1
export PREFILL_PORT=8100
export DECODE_PORT=8200
export DECODE_SIDECAR_PORT=8300

# Paths
export PROJECT_DIR=/project
export CONFIG_DIR=/run/configs
export OUTPUT_DIR=/output/golden
export LAUNCHER_DIR=/output/golden/launcher
export LOG_DIR=/run/logs
export PRL_ATTEMPT_CONFIG_DIR=$CONFIG_DIR
export PRL_ATTEMPT_LOG_DIR=$LOG_DIR
mkdir -p $LOG_DIR/trainer $LOG_DIR/inference
ln -sfn trainer/node_0.log $LOG_DIR/trainer.log
ln -sfn inference/node_0.log $LOG_DIR/inference.log

# Clean exits do not trigger srun --kill-on-bad-exit. Each terminal component
# publishes its successful exit through the shared launcher directory. Every node
# uses the same completion condition before it tears down support processes.
export TRAINER_DONE=$LAUNCHER_DIR/.trainer.done
export ORCHESTRATOR_DONE=$LAUNCHER_DIR/.orchestrator.done
mkdir -p "$LAUNCHER_DIR"
rm -f "$TRAINER_DONE" "$ORCHESTRATOR_DONE"

export PRL_RUN_ID=${PRL_RUN_ID:-$(python3 -c "import uuid; print(uuid.uuid4().hex)")}
export PRL_RUN_NAME=golden
export WANDB_RUN_ID=$PRL_RUN_ID
export WANDB_SHARED_MODE=1

# Networking
HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
TRAIN_HOSTS=( ${HOSTNAMES[@]:$NUM_INFER_NODES:$SLURM_JOB_NUM_NODES} )
export HOSTNAMES_STR="${HOSTNAMES[*]}"

INFER_HOSTS=( ${HOSTNAMES[@]:0:$NUM_INFER_NODES} )

# Single global router on inference node 0 fronts every per-rank endpoint;
# admin URLs bypass it and hit each rank directly.
INFER_URLS="http://${INFER_HOSTS[0]}:${ROUTER_PORT}/v1"
ADMIN_URLS=""
ROUTER_ARGS=""
# External-LB: one prefill/decode endpoint per DP rank (PORT + k) on each node.
for ((r=0; r<NUM_INFER_REPLICAS; r++)); do
    replica_base=$((r * NODES_PER_INFER_REPLICA))
    for ((p=0; p<NUM_PREFILL_NODES; p++)); do
        host="${INFER_HOSTS[$((replica_base + p))]}"
        for ((k=0; k<INFERENCE_DP_LOCAL; k++)); do
            ADMIN_URLS="${ADMIN_URLS:+$ADMIN_URLS }http://${host}:$((PREFILL_PORT + k))/v1"
            ROUTER_ARGS="$ROUTER_ARGS --prefill http://${host}:$((PREFILL_PORT + k))"
        done
    done
    for ((d=0; d<NUM_DECODE_NODES; d++)); do
        host="${INFER_HOSTS[$((replica_base + NUM_PREFILL_NODES + d))]}"
        for ((k=0; k<INFERENCE_DP_LOCAL; k++)); do
            ADMIN_URLS="${ADMIN_URLS:+$ADMIN_URLS }http://${host}:$((DECODE_PORT + k))/v1"
            ROUTER_ARGS="$ROUTER_ARGS --decode http://${host}:$((DECODE_PORT + k))"
        done
    done
done
export INFER_URLS
export ADMIN_URLS
export ROUTER_ARGS
echo "INFER_HOSTS=${INFER_HOSTS[*]}"
echo "INFER_URLS=${INFER_URLS}"
echo "ADMIN_URLS=${ADMIN_URLS}"

echo "HOSTNAMES=${HOSTNAMES[*]}"
echo "TRAIN_HOSTS=${TRAIN_HOSTS[*]}"

export MASTER_ADDR="${HOSTNAMES[$NUM_INFER_NODES]}"
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"


# The orchestrator runs on trainer rank 0's node (SLURM_PROCID == NUM_INFER_NODES) by
# default; with orchestrator_on_inference it moves to the last inference node (a decode
# node) to keep its rollout buffer + env workers off host-RAM-saturated trainer nodes.
# The ZMQ rollout transport binds in the orchestrator process, so both sides must
# address the orchestrator's node.
ORCH_PROCID=$NUM_INFER_NODES
export ORCH_PROCID
export ORCH_ADDR="${HOSTNAMES[$ORCH_PROCID]}"
echo "ORCH_ADDR=${ORCH_ADDR}"

# Setup environment
cd $PROJECT_DIR
[ -f .env ] && source .env
source .venv/bin/activate

# Shared (NFS) venv: one sync on the batch node populates it for all nodes.
uv sync --all-extras --all-packages




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

# Run RL
# --kill-on-bad-exit=1 propagates a non-zero exit on any node to all other
# tasks so a crashed component terminates the SLURM job instead of leaving the
# surviving ranks holding nodes.
#
# The body goes to bash on stdin through a quoted heredoc instead of a
# single-quoted `bash -c` argument: srun broadcasts stdin to every task, and the
# quoted delimiter passes the script through verbatim, so apostrophes and quotes
# (including in the shared _launch_*.sh.j2 includes) need no escaping. Nothing in
# the body reads stdin, which would otherwise eat the rest of the script.
srun --kill-on-bad-exit=1 bash -s <<'LAUNCH_SH'
    # pipefail ensures `cmd | tee log` propagates `cmd` failures, and the
    # wait/cleanup block at the end of this script surfaces background
    # process crashes (router, orchestrator) as task failures.
    set -o pipefail

    # Source environment
    cd $PROJECT_DIR
    [ -f .env ] && source .env
    source .venv/bin/activate
    # Do not sync to avoid conflicts with lockfile

    # Higher ulimit
    ulimit -n 65536

    # Infiniband setup
    IB_HCA=$(ibv_devinfo | sed -n -e "/hca_id/p" -e "/link_layer:/p" | grep -B1 InfiniBand | grep hca_id | sed -e "s/^hca_id://g" | tr -d "[[:blank:]]" |paste -sd,)
    export NCCL_IB_HCA=$IB_HCA



if [ "$SLURM_PROCID" -lt "$NUM_INFER_NODES" ]; then
    INFER_NODE_RANK=$SLURM_PROCID
    # Inference env vars: launcher defaults + [env_vars] + [inference.env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
    export VLLM_ENGINE_READY_TIMEOUT_S="4200"
    export UCX_TLS="all"
    LOCAL_IP=$(hostname -I | awk "{print \$1}")
    ulimit -l unlimited
# ---- Mooncake distributed store (shared pool across inference nodes) ----
    MC_DIR="$OUTPUT_DIR/mooncake/node_${INFER_NODE_RANK}"
    mkdir -p "$MC_DIR"
    export PYTHONHASHSEED=0
    export MOONCAKE_CONFIG_PATH="$MC_DIR/config.json"
    # The head inference node (first in the nodelist) hosts the one master + metadata server.
    MC_HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
    MC_MASTER="$MC_HEAD:50051"
    MC_METADATA="http://$MC_HEAD:8080/metadata"
    if [ "$SLURMD_NODENAME" = "$MC_HEAD" ]; then
        # default_kv_lease_ttl raised from 5s: KV blocks were expiring before the decode side
        # reused them ("invalid KV blocks ... may have expired"), the real non-NIC config issue.
        mooncake_master -rpc_port=50051 -enable_http_metadata_server -http_metadata_server_host=0.0.0.0 -http_metadata_server_port=8080 -default_kv_lease_ttl=3600000 > "$MC_DIR/master.log" 2>&1 &
    fi
    # Every node waits for the shared master + metadata, then contributes its segment.
    for _ in $(seq 1 120); do timeout 1 bash -c "echo > /dev/tcp/$MC_HEAD/50051" 2>/dev/null && break; sleep 1; done
    for _ in $(seq 1 120); do timeout 1 bash -c "echo > /dev/tcp/$MC_HEAD/8080" 2>/dev/null && break; sleep 1; done
    mooncake_client -host=$LOCAL_IP -port=50052 -master_server_address=$MC_MASTER -metadata_server=$MC_METADATA -protocol=rdma -global_segment_size=1000000000000 -device_names=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11 > "$MC_DIR/client.log" 2>&1 &
    for _ in $(seq 1 60); do timeout 1 bash -c "echo > /dev/tcp/$LOCAL_IP/50052" 2>/dev/null && break; sleep 1; done
    cat > "$MOONCAKE_CONFIG_PATH" <<MOONCAKE_JSON
{"mode":"standalone-store","global_segment_size":0,"local_buffer_size":4294967296,"protocol":"rdma","device_name":"mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11","master_server_address":"$MC_MASTER","metadata_server":"$MC_METADATA","enable_offload":false}
MOONCAKE_JSON
    echo "[mooncake] node ${INFER_NODE_RANK} ($SLURMD_NODENAME) joined shared pool: master=$MC_MASTER segment=$LOCAL_IP:50052"


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
    echo "Starting llm-d router (EPP+Envoy) on $LOCAL_IP:$port" | tee "$log"
    export PATH="$PROJECT_DIR/third_party/llmd/bin:$PATH"
    local LLMD_DIR="$LOG_DIR/inference/llmd"
    mkdir -p "$LLMD_DIR"
    read -ra HOSTNAMES <<< "$HOSTNAMES_STR"
    cat > "$LLMD_DIR/epp.yaml" <<LLMD_EOF

apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - name: file-disc
    type: file-discovery
    parameters:
      path: $LLMD_DIR/endpoints.yaml
      watchFile: true
  - name: vllmhttp-parser
    type: vllmhttp-parser
  - name: pd-decider
    type: prefix-based-pd-decider
    parameters:
      nonCachedTokens: 128
  - name: profile-handler
    type: disagg-profile-handler
    parameters:
      deciders:
        prefill: pd-decider
  - type: prefill-filter
  - type: decode-filter
  - type: prefix-cache-scorer
  - type: queue-scorer
  - type: active-request-scorer
  - type: kv-cache-utilization-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: prefill-filter
      - pluginRef: prefix-cache-scorer
        weight: 3.0
      - pluginRef: queue-scorer
        weight: 4.0
      - pluginRef: active-request-scorer
        weight: 5.0
      - pluginRef: max-score-picker
  - name: decode
    plugins:
      - pluginRef: decode-filter
      - pluginRef: prefix-cache-scorer
        weight: 3.0
      - pluginRef: active-request-scorer
        weight: 5.0
      - pluginRef: kv-cache-utilization-scorer
        weight: 4.0
      - pluginRef: queue-scorer
        weight: 3.0
      - pluginRef: max-score-picker
requestHandler:
  parser:
    pluginRef: vllmhttp-parser
dataLayer:
  discovery:
    pluginRef: file-disc

LLMD_EOF
    cat > "$LLMD_DIR/endpoints.yaml" <<LLMD_EOF

endpoints:
  - name: prefill-0-rank-0
    address: __LLMD_ADDR_0__
    port: "8100"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: prefill
  - name: prefill-0-rank-1
    address: __LLMD_ADDR_0__
    port: "8101"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: prefill
  - name: prefill-0-rank-2
    address: __LLMD_ADDR_0__
    port: "8102"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: prefill
  - name: prefill-0-rank-3
    address: __LLMD_ADDR_0__
    port: "8103"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: prefill
  - name: prefill-0-rank-4
    address: __LLMD_ADDR_0__
    port: "8104"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: prefill
  - name: prefill-0-rank-5
    address: __LLMD_ADDR_0__
    port: "8105"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: prefill
  - name: prefill-0-rank-6
    address: __LLMD_ADDR_0__
    port: "8106"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: prefill
  - name: prefill-0-rank-7
    address: __LLMD_ADDR_0__
    port: "8107"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: prefill
  - name: decode-0-rank-0
    address: __LLMD_ADDR_1__
    port: "8300"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: decode
  - name: decode-0-rank-1
    address: __LLMD_ADDR_1__
    port: "8301"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: decode
  - name: decode-0-rank-2
    address: __LLMD_ADDR_1__
    port: "8302"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: decode
  - name: decode-0-rank-3
    address: __LLMD_ADDR_1__
    port: "8303"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: decode
  - name: decode-0-rank-4
    address: __LLMD_ADDR_1__
    port: "8304"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: decode
  - name: decode-0-rank-5
    address: __LLMD_ADDR_1__
    port: "8305"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: decode
  - name: decode-0-rank-6
    address: __LLMD_ADDR_1__
    port: "8306"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: decode
  - name: decode-0-rank-7
    address: __LLMD_ADDR_1__
    port: "8307"
    namespace: default
    labels:
      llm-d.ai/pool: prime-rl
      llm-d.ai/role: decode

LLMD_EOF
    local node=0 ip reps=${NUM_INFER_REPLICAS:-1} span=${NODES_PER_INFER_REPLICA:-$((NUM_PREFILL_NODES + NUM_DECODE_NODES))}
    for ((rr=0; rr<reps; rr++)); do
        for ((p=0; p<NUM_PREFILL_NODES; p++)); do
            ip=$(getent hosts "${HOSTNAMES[$((rr * span + p))]}" | awk "{print \$1}")
            sed -i "s|__LLMD_ADDR_${node}__|${ip}|g" "$LLMD_DIR/endpoints.yaml"; node=$((node + 1))
        done
    done
    for ((rr=0; rr<reps; rr++)); do
        for ((d=0; d<NUM_DECODE_NODES; d++)); do
            ip=$(getent hosts "${HOSTNAMES[$((rr * span + NUM_PREFILL_NODES + d))]}" | awk "{print \$1}")
            sed -i "s|__LLMD_ADDR_${node}__|${ip}|g" "$LLMD_DIR/endpoints.yaml"; node=$((node + 1))
        done
    done
    cat > "$LLMD_DIR/envoy.yaml" <<LLMD_EOF

admin:
  address: { socket_address: { address: 127.0.0.1, port_value: 9901 } }
static_resources:
  listeners:
    - name: prime_rl
      address: { socket_address: { address: 0.0.0.0, port_value: 8000 } }
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: prime_rl
                codec_type: AUTO
                stream_idle_timeout: 0s
                request_timeout: 0s
                route_config:
                  virtual_hosts:
                    - name: prime_rl
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/v1/" }
                          route: { cluster: backend_cluster, timeout: 0s, idle_timeout: 86400s }
                        - match: { prefix: "/inference/v1/" }
                          route: { cluster: backend_cluster, timeout: 0s, idle_timeout: 86400s }
                http_filters:
                  - name: envoy.filters.http.ext_proc
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
                      grpc_service:
                        envoy_grpc: { cluster_name: epp_cluster }
                        timeout: 10s
                      processing_mode:
                        request_header_mode: SEND
                        response_header_mode: SEND
                        request_body_mode: FULL_DUPLEX_STREAMED
                        response_body_mode: FULL_DUPLEX_STREAMED
                        request_trailer_mode: SEND
                        response_trailer_mode: SEND
                      message_timeout: 1000s
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
                      suppress_envoy_headers: true
  clusters:
    - name: backend_cluster
      type: ORIGINAL_DST
      lb_policy: CLUSTER_PROVIDED
      original_dst_lb_config:
        use_http_header: true
        http_header_name: x-gateway-destination-endpoint
      connect_timeout: 5s
      # Default circuit breakers (~1024) overflow at high inflight concurrency (>=1536),
      # causing "reset reason: overflow" 500s. Raise far above the max concurrent rollouts.
      circuit_breakers:
        thresholds:
          - priority: DEFAULT
            max_connections: 100000
            max_pending_requests: 100000
            max_requests: 100000
            max_retries: 1000
          - priority: HIGH
            max_connections: 100000
            max_pending_requests: 100000
            max_requests: 100000
            max_retries: 1000
    - name: epp_cluster
      type: STRICT_DNS
      typed_extension_protocol_options:
        envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
          "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
          explicit_http_config:
            http2_protocol_options: {}
      load_assignment:
        cluster_name: epp_cluster
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address: { socket_address: { address: 127.0.0.1, port_value: 9002 } }
      connect_timeout: 5s
      # ext_proc (EPP) connection pool overflows at high concurrency without this — the
      # gRPC ext_proc calls are 1:1 with inflight requests, so cap well above max rollouts.
      circuit_breakers:
        thresholds:
          - priority: DEFAULT
            max_connections: 100000
            max_pending_requests: 100000
            max_requests: 100000
            max_retries: 1000
          - priority: HIGH
            max_connections: 100000
            max_pending_requests: 100000
            max_requests: 100000
            max_retries: 1000

LLMD_EOF
    # grpc-health-port avoids 9003: mooncake_master runs its metrics server on 9003 and
    # shares node 0 with the EPP, so 9003 would fail to bind and kill the EPP.
    epp --pool-name prime-rl --pool-namespace slurm \
        --config-file "$LLMD_DIR/epp.yaml" \
        --grpc-port 9002 --grpc-health-port 9013 --metrics-port 9090 \
        --secure-serving=false --metrics-endpoint-auth=false --tracing=false \
        >> "$log" 2>&1 &
    envoy -c "$LLMD_DIR/envoy.yaml" >> "$log" 2>&1 &
}


# PD disaggregated: NIXL KV transfer + role-based inference
    read -ra HOSTNAMES <<< "$HOSTNAMES_STR"

    # Bind gloo to the interface that owns LOCAL_IP (the rendezvous IP) — NIC
    # names vary across clusters (bond0, eth0, ...). If the lookup fails, leave
    # GLOO_SOCKET_IFNAME unset so gloo falls back to its own detection.
    GLOO_IFNAME=$(ip -o -4 addr show to "$LOCAL_IP" 2>/dev/null | awk "{print \$2; exit}")
    if [ -n "$GLOO_IFNAME" ]; then export GLOO_SOCKET_IFNAME=$GLOO_IFNAME; fi
    # Enumerate RDMA NICs for UCX/NIXL from sysfs (ibv_devices is not on the job
    # PATH). Prefer InfiniBand ports and fall back to RoCE: "all" would also pick
    # up non-functional devices (e.g. an Ethernet bond UCX cannot open), which
    # fails NIXL backend creation. Hosts without RDMA devices keep "all" (TCP).
    rdma_ports() {
        for port in /sys/class/infiniband/*/ports/*; do
            [ "$(cat "$port/link_layer" 2>/dev/null)" = "$1" ] || continue
            grep -q ACTIVE "$port/state" 2>/dev/null || continue
            dev=${port%/ports/*}
            echo "$(basename "$dev"):$(basename "$port")"
        done | paste -sd,
    }
    UCX_NET_DEVICES=$(rdma_ports InfiniBand)
    [ -n "$UCX_NET_DEVICES" ] || UCX_NET_DEVICES=$(rdma_ports Ethernet)
    export UCX_NET_DEVICES=${UCX_NET_DEVICES:-all}
    export LD_LIBRARY_PATH="$PROJECT_DIR/third_party/ucx/lib:$PROJECT_DIR/third_party/ucx/lib/ucx:${LD_LIBRARY_PATH:-}"
    export VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5600

    # The kv_transfer_config (NIXL transfer + any KV offload connector) is built in
    # InferenceConfig.to_namespace() from the per-node config, so it is no longer assembled here.

    DECODE_COMPILE_CFG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'

    REPLICA_BASE=$((REPLICA_IDX * NODES_PER_INFER_REPLICA))

    if [ "$RANK_IN_REPLICA" -lt "$NUM_PREFILL_NODES" ]; then
        ROLE="prefill"
        SUB_REPLICA=$((RANK_IN_REPLICA / NODES_PER_PREFILL_REPLICA))
        ROLE_RANK=$((RANK_IN_REPLICA % NODES_PER_PREFILL_REPLICA))
        DP=$((NODES_PER_PREFILL_REPLICA * INFERENCE_DP_LOCAL))
        PORT=$PREFILL_PORT
        START_RANK=$((ROLE_RANK * INFERENCE_DP_LOCAL))
        HEAD_HOST="${HOSTNAMES[$((REPLICA_BASE + SUB_REPLICA * NODES_PER_PREFILL_REPLICA))]}"
        # vLLM >= 0.29 auto-enables breakable CUDA graphs; capturing the DeepEP
        # high-throughput internode dispatch then fails ("timeout (dispatch CPU)").
        export VLLM_USE_BREAKABLE_CUDAGRAPH=0

        export VLLM_USE_BREAKABLE_CUDAGRAPH="0"

        export VLLM_ENABLE_MOE_DP_CHUNK="0"

        export VLLM_DEEP_GEMM_WARMUP="skip"

        export UCX_RCACHE_MAX_UNRELEASED="1024"

        ROLE_EXTRA="\"all2all_backend\": \"deepep_high_throughput\", \"enable_dbo\": true"
    else
        export UCX_NET_DEVICES=mlx5_0:1
        ROLE="decode"
        RANK_IN_DECODE=$((RANK_IN_REPLICA - NUM_PREFILL_NODES))
        SUB_REPLICA=$((RANK_IN_DECODE / NODES_PER_DECODE_REPLICA))
        ROLE_RANK=$((RANK_IN_DECODE % NODES_PER_DECODE_REPLICA))
        DP=$((NODES_PER_DECODE_REPLICA * INFERENCE_DP_LOCAL))
        PORT=$DECODE_PORT
        START_RANK=$((ROLE_RANK * INFERENCE_DP_LOCAL))
        HEAD_HOST="${HOSTNAMES[$((REPLICA_BASE + NUM_PREFILL_NODES + SUB_REPLICA * NODES_PER_DECODE_REPLICA))]}"

        export VLLM_DEEP_GEMM_WARMUP="skip"

        export UCX_RCACHE_MAX_UNRELEASED="1024"

        ROLE_EXTRA="\"all2all_backend\": \"deepep_low_latency\", \"compilation_config\": $DECODE_COMPILE_CFG, \"max_num_batched_tokens\": 96, \"max_num_seqs\": 96"
    fi

    echo "ROLE=$ROLE REPLICA=$REPLICA_IDX SUB_REPLICA=$SUB_REPLICA ROLE_RANK=$ROLE_RANK DP=$DP PORT=$PORT" \
        | tee -a $LOG_DIR/inference/node_${INFER_NODE_RANK}.log

    # Start the single global router on inference node 0 (PD mode, external LB across all per-rank endpoints)
    if [ "$INFER_NODE_RANK" -eq 0 ]; then
        launch_router pd "$ROUTER_ARGS" "$ROUTER_PORT" "" "$LOG_DIR/inference/router.log"
    fi
    # pd-sidecar on each decode node: EPP routes decode -> sidecar (DECODE_SIDECAR_PORT
    # + rank) -> vLLM decode (DECODE_PORT + rank) after orchestrating remote prefill.
    # Every decode node fronts its own local ranks, so launch on each (not just the
    # replica head), or decode nodes beyond the head get advertised but unserved endpoints.
    if [ "$ROLE" = "decode" ]; then
        SIDECAR_LOG="$LOG_DIR/inference/sidecar_${INFER_NODE_RANK}.log"
        echo "Starting pd-sidecar on $LOCAL_IP:$DECODE_SIDECAR_PORT(+rank) -> vLLM $DECODE_PORT(+rank)" | tee $SIDECAR_LOG
        export PATH="$PROJECT_DIR/third_party/llmd/bin:$PATH"
        pd-sidecar \
            --port $DECODE_SIDECAR_PORT \
            --vllm-port $DECODE_PORT \
            --kv-connector nixlv2 \
            --secure-proxy=false \
            --enable-ssrf-protection=false \
            --inference-pool prime-rl \
            --data-parallel-size $INFERENCE_DP_LOCAL \
            >> $SIDECAR_LOG 2>&1 &
    fi

    # External-LB: one vLLM API server per local DP rank (PORT + k), each pinned to its TP
    # slice of GPUs with a unique NIXL side-channel port. Each role is its own external-LB DP
    # group rooted on the role-replica head (HEAD_HOST); the router balances across the per-rank
    # endpoints. The kv_transfer_config is built in InferenceConfig.to_namespace() from the config.
    if [ "$ROLE_RANK" -eq 0 ]; then DP_ADDR=$LOCAL_IP; else DP_ADDR=$HEAD_HOST; fi
    for k in $(seq 0 $((INFERENCE_DP_LOCAL - 1))); do
        GLOBAL_RANK=$((START_RANK + k))
        RANK_GPUS=$(seq -s, $((k * INFERENCE_TP)) $((k * INFERENCE_TP + INFERENCE_TP - 1)))
        RANK_EXTRA="{\"data_parallel_address\": \"$DP_ADDR\", \"data_parallel_rank\": $GLOBAL_RANK, $ROLE_EXTRA}"
        launch_inference_rank "$((PORT + k))" "$RANK_GPUS" "$DP" "$INFERENCE_DATA_PARALLEL_RPC_PORT" "$RANK_EXTRA" "$((5600 + k))" "$(rank_log "$INFER_NODE_RANK" "$k")"
    done

else
    TRAIN_NODE_RANK=$((SLURM_PROCID - NUM_INFER_NODES))

    # Trainer env vars: launcher defaults + [env_vars] + [trainer.env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    echo $HOSTNAMES_STR | tee $LOG_DIR/trainer/node_${TRAIN_NODE_RANK}.log

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
            -m prime_rl.trainer.rl.train \
            @ $CONFIG_DIR/trainer.json \
            --rollout_transport.host $ORCH_ADDR \
            2>&1 | sed -u 's/^\[[a-zA-Z]*[0-9]*\]://' | tee -a $LOG_DIR/trainer/node_${TRAIN_NODE_RANK}.log
        COMPONENT_CODE=$?
        if [ "$COMPONENT_CODE" -eq 0 ] && [ "$TRAIN_NODE_RANK" -eq 0 ]; then
            touch "$TRAINER_DONE" || exit $?
        fi
        exit "$COMPONENT_CODE"
    ) &
    fi

# Launch the orchestrator on the node ORCH_ADDR points at (see the ORCH_PROCID
# computation above). Top-level (after the role if/else) so the wait -n teardown
# covers it.
if [ "$SLURM_PROCID" -eq "$ORCH_PROCID" ]; then
    ORCHESTRATOR_ARGS="@ $CONFIG_DIR/orchestrator.json"
    ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --model.client.base-url $INFER_URLS"
    ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --model.client.admin-base-url $ADMIN_URLS"
    # Orchestrator env vars from [env_vars] + [orchestrator.env_vars]
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export PYTHONUNBUFFERED="1"
    export OMP_NUM_THREADS="1"
    export GIT_LFS_SKIP_SMUDGE="1"

    # Launch one env server per source next to the orchestrator, which connects to each
    # source on this node; each publishes the loopback address it bound, polling until they are
    # up — the servers and the orchestrator start in parallel.
    # NOTE: keep this block free of single quotes. The whole srun body is one
    # single-quoted bash -c string, so a stray quote closes it early and the rest of
    # the script fails to parse.
    mkdir -p $LOG_DIR/envs/train
    uv run env-server @ $CONFIG_DIR/envs/train/r2e.json \
        > $LOG_DIR/envs/train/r2e.log 2>&1 &

    (
        ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --weight_broadcast.host $MASTER_ADDR"
        ORCHESTRATOR_ARGS="$ORCHESTRATOR_ARGS --rollout_transport.host $ORCH_ADDR"
        WANDB_SHARED_LABEL=orchestrator uv run orchestrator $ORCHESTRATOR_ARGS \
            2>&1 | tee $LOG_DIR/orchestrator.log
        COMPONENT_CODE=$?
        if [ "$COMPONENT_CODE" -eq 0 ]; then
            touch "$ORCHESTRATOR_DONE" || exit $?
        fi
        exit "$COMPONENT_CODE"
    ) &
fi


# Every node watches the terminal components for successful completion.
# Inference runs stop after both the trainer and orchestrator finish.
( until [ -f "$TRAINER_DONE" ] && [ -f "$ORCHESTRATOR_DONE" ]; do sleep 10; done ) &
COMPLETION_PID=$!

# A clean component exit is not enough to stop the run. Keep waiting for the
# shared completion condition. Propagate any failure through
# --kill-on-bad-exit=1 so every node tears down.
while true; do
    wait -n -p FINISHED_PID
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -ne 0 ] || [ "${FINISHED_PID:-}" = "$COMPLETION_PID" ]; then
        break
    fi
done

REMAINING_PIDS=$(jobs -p)
# Cross-node teardown is driven by `srun --kill-on-bad-exit=1`, which reaps peer
# tasks only once a task *exits* non-zero. So on a non-zero exit (crash, SIGTERM,
# wall-time) we stay alive for the grace period and signal nothing: this lets an
# in-flight trainer checkpoint on this node finish, and — because the task has
# not exited yet — keeps --kill-on-bad-exit from reaping checkpointing trainers
# on peer nodes. We then propagate the exit code so the allocation tears down. A
# clean (zero) exit cannot trip --kill-on-bad-exit, so we tear down immediately.
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[$(hostname)] component exited with code $EXIT_CODE - waiting up to 3600s for in-flight checkpoints to flush before teardown"
    sleep 3600
fi
echo "[$(hostname)] tearing down (exit code $EXIT_CODE) - terminating remaining processes"
for pid in $REMAINING_PIDS; do
    kill -TERM $pid 2>/dev/null || true
done
exit $EXIT_CODE
LAUNCH_SH
