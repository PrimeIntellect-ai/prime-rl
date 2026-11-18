#!/bin/bash
# Hand-launched 1-node P/D: prefill DP$NP on GPUs [0, NP), decode DP$ND on GPUs [NP, NP+ND),
# NIXL KV transfer, vllm-router in PD mode on :8000. Mirrors inference.sbatch.j2's per-rank launch.
#
#   pd1node.sh <config_dir> <log_dir> <prefill_a2a> <decode_a2a> [NP] [ND] [decode_cudagraph_mode]
set -uo pipefail
CONFIG_DIR=$1
LOG_DIR=$2
P_A2A=$3
D_A2A=$4
NP=${5:-4}
ND=${6:-4}
D_CG=${7:-FULL_DECODE_ONLY}
PROJECT_DIR=${PROJECT_DIR:-/home/phi/lib/prime-rl-pd}

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"
source .venv/bin/activate
ulimit -l unlimited

LOCAL_IP=$(hostname -I | awk '{print $1}')
GLOO_IFNAME=$(ip -o -4 addr show to "$LOCAL_IP" 2>/dev/null | awk '{print $2; exit}')
[ -n "$GLOO_IFNAME" ] && export GLOO_SOCKET_IFNAME=$GLOO_IFNAME
rdma_ports() {
    for port in /sys/class/infiniband/*/ports/*; do
        [ "$(cat "$port/link_layer" 2>/dev/null)" = "$1" ] || continue
        grep -q ACTIVE "$port/state" 2>/dev/null || continue
        dev=${port%/ports/*}
        echo "$(basename "$dev"):$(basename "$port")"
    done | paste -sd,
}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-$(rdma_ports InfiniBand)}
[ -d "$PROJECT_DIR/third_party/ucx/lib" ] && export LD_LIBRARY_PATH="$PROJECT_DIR/third_party/ucx/lib:$PROJECT_DIR/third_party/ucx/lib/ucx:${LD_LIBRARY_PATH:-}"
export VLLM_NIXL_SIDE_CHANNEL_HOST=$LOCAL_IP
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/home/phi/.cache/vllm-flashinfer-autotune
export PYTHONUNBUFFERED=1

ROLES=${ROLES:-prefill decode}  # subset to (re)launch; the router starts only with both
if [ "$ROLES" = "prefill decode" ]; then
    pkill -9 -f "[v]llm-router" 2>/dev/null
    pkill -9 -f "[i]nference @" 2>/dev/null
    pkill -9 -f "VLLM::" 2>/dev/null
    sleep 3
    rm -rf /tmp/vllm-rpc-pd1node-* 2>/dev/null
fi

launch() {  # role k gpu port dp rpc nixl extra_json
    local role=$1 k=$2 gpu=$3 port=$4 dp=$5 rpc=$6 nixl=$7 extra=$8
    local rpcbase="/tmp/vllm-rpc-pd1node-$port"
    mkdir -p "$rpcbase"
    printf '%s' "$extra" > "$rpcbase/vllm-overrides.json"
    CUDA_VISIBLE_DEVICES=$gpu VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl VLLM_RPC_BASE_PATH=$rpcbase \
        uv run --no-sync inference @ "$CONFIG_DIR/inference-$role.json" \
        --server.host 0.0.0.0 --server.port "$port" \
        --vllm.data-parallel-size "$dp" --vllm.data-parallel-size-local 1 --vllm.api-server-count 1 \
        --vllm.data-parallel-rpc-port "$rpc" --vllm @ "$rpcbase/vllm-overrides.json" \
        > "$LOG_DIR/${role}_$k.log" 2>&1 &
}

ROUTER_ARGS=""
for ((k=0; k<NP; k++)); do
    [[ " $ROLES " == *" prefill "* ]] && launch prefill $k $k $((8100 + k)) $NP 13345 $((5600 + k)) \
        "{\"data_parallel_address\": \"$LOCAL_IP\", \"data_parallel_rank\": $k, \"all2all_backend\": \"$P_A2A\"}"
    ROUTER_ARGS="$ROUTER_ARGS --prefill http://$LOCAL_IP:$((8100 + k))"
done
for ((k=0; k<ND; k++)); do
    [[ " $ROLES " == *" decode "* ]] && launch decode $k $((NP + k)) $((8200 + k)) $ND 13445 $((5700 + k)) \
        "{\"data_parallel_address\": \"$LOCAL_IP\", \"data_parallel_rank\": $k, \"all2all_backend\": \"$D_A2A\", \"compilation_config\": {\"cudagraph_mode\": \"$D_CG\"}}"
    ROUTER_ARGS="$ROUTER_ARGS --decode http://$LOCAL_IP:$((8200 + k))"
done

[ "$ROLES" = "prefill decode" ] && vllm-router --policy sticky_least_loaded --host 0.0.0.0 --port 8000 \
    --request-id-headers x-session-id --request-timeout-secs 14400 --prometheus-port 29000 \
    --worker-startup-timeout-secs 4200 --log-level debug \
    --vllm-pd-disaggregation $ROUTER_ARGS > "$LOG_DIR/router.log" 2>&1 &

echo "launched: prefill=$P_A2A x$NP decode=$D_A2A x$ND ($D_CG) on $LOCAL_IP; logs in $LOG_DIR"
wait
