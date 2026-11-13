#!/bin/bash
# Usage: node.sh (one task per node under srun; reads TRAIN_ARM_DIR, INFER_ARM_DIR, RUN_DIR, CONFIG_DIR, TRAIN_HOST,
# INFER_HOST, MASTER_PORT, MAX_STEPS, RUN_TIMEOUT_S, TEARDOWN_TIMEOUT_S)
# Mirrors the non-disaggregated branch of multi_node_rl.sbatch.j2 for one trainer node and one inference node, with
# per-component exit records, a post-final-step hang watchdog on the trainer node and a timed SIGTERM teardown of the
# inference node once the trainer node has finished.
set -uo pipefail

harness_dir=$(dirname "$(readlink -f "$0")")
bench=$RUN_DIR/bench
export EXIT_DIR=$bench/exits
abort_file=$bench/abort
train_finished_file=$bench/trainer_node.finished
mkdir -p "$EXIT_DIR"

if [ "$(hostname)" = "$TRAIN_HOST" ]; then
    role=train
    arm_dir=$TRAIN_ARM_DIR
else
    role=infer
    arm_dir=$INFER_ARM_DIR
fi
node_info=$bench/node_${role}.txt

cd "$arm_dir" || exit 1
source .venv/bin/activate
ulimit -n 65536

IB_HCA=$(ibv_devinfo 2>/dev/null | sed -n -e "/hca_id/p" -e "/link_layer:/p" | grep -B1 InfiniBand | grep hca_id | sed -e "s/^hca_id://g" | tr -d "[[:blank:]]" | paste -sd,)
[ -n "$IB_HCA" ] && export NCCL_IB_HCA=$IB_HCA
export NCCL_DEBUG=VERSION
export PYTHONPATH=$arm_dir/deps/verifiers/environments/reverse_text

export PRL_RUN_NAME=$(basename "$RUN_DIR")
export PRL_RUN_ID=$PRL_RUN_NAME
export WANDB_SHARED_MODE=1
export WANDB_RUN_ID=$PRL_RUN_ID
export PRL_ATTEMPT_CONFIG_DIR=$CONFIG_DIR
export PRL_ATTEMPT_LOG_DIR=$RUN_DIR/logs/attempt_1
mkdir -p "$PRL_ATTEMPT_LOG_DIR"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="1"
export GIT_LFS_SKIP_SMUDGE="1"

cache_root=/tmp/nccl-nic-rl-$PRL_RUN_NAME
rm -rf "$cache_root"
export TRITON_CACHE_DIR=$cache_root/triton
export TORCHINDUCTOR_CACHE_DIR=$cache_root/inductor
export VLLM_CACHE_ROOT=$cache_root/vllm
export FLASHINFER_WORKSPACE_BASE=$cache_root/flashinfer
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

nccl_version=$(python -c 'import ctypes, torch; v = ctypes.c_int(); ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(v)); print(v.value)' 2>&1 | tail -1)
{
    echo "host $(hostname) role $role arm $arm_dir commit $(git -C "$arm_dir" rev-parse --short=9 HEAD) libnccl $nccl_version"
    echo "NCCL_IB_HCA=${NCCL_IB_HCA:-} cache_root=$cache_root"
    echo "gpu_mem_used_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd,)"
} > "$node_info"

backend_ports=(8100 8101 8102 8103)
admin_urls=""
router_workers=()
for port in "${backend_ports[@]}"; do
    admin_urls="${admin_urls:+$admin_urls }http://$INFER_HOST:$port/v1"
    router_workers+=("http://$INFER_HOST:$port")
done

exited() {
    grep -q "^exit " "$EXIT_DIR/$1.txt" 2>/dev/null
}

exit_rc() {
    sed -n "s/^exit [0-9.]* rc //p" "$EXIT_DIR/$1.txt" 2>/dev/null
}

comp_pid() {
    sed -n "s/^start .* pid //p" "$EXIT_DIR/$1.txt" 2>/dev/null
}

proc_state() {
    sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f1
}

descendants() {
    local child
    for child in $(ps -o pid= --ppid "$1" 2>/dev/null); do
        echo "$child"
        descendants "$child"
    done
}

run_train_node() {
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCH_FR_BUFFER_SIZE="2000"
    export TORCH_NCCL_DUMP_ON_TIMEOUT="1"
    export TORCH_FR_DUMP_TEMP_FILE="/tmp/nccl_trace_$(id -u)_rank_"
    export RANK_LOG_DIR=$bench/ranks
    mkdir -p "$RANK_LOG_DIR"

    "$harness_dir/comp.sh" trainer_torchrun "$bench/torchrun.log" \
        torchrun \
        --role=trainer \
        --nnodes=1 \
        --nproc-per-node=8 \
        --node-rank=0 \
        --rdzv-endpoint="$TRAIN_HOST:$MASTER_PORT" \
        --rdzv-id="$PRL_RUN_NAME" \
        --no-python \
        "$harness_dir/rank.sh" &

    # shellcheck disable=SC2086
    "$harness_dir/comp.sh" orchestrator "$bench/orchestrator.log" \
        orchestrator @ "$CONFIG_DIR/orchestrator.json" \
        --model.client.base-url "http://$INFER_HOST:8000/v1" \
        --model.client.admin-base-url $admin_urls \
        --weight_broadcast.host "$TRAIN_HOST" \
        --rollout_transport.host "$TRAIN_HOST" &

    local start_time=$EPOCHSECONDS last_step_seen="" verdict="" now
    while true; do
        sleep 2
        now=$EPOCHSECONDS
        if exited trainer_torchrun && exited orchestrator; then
            break
        fi
        if [ -z "$last_step_seen" ] && grep "Step $MAX_STEPS |" "$RANK_LOG_DIR/rank_0.log" 2>/dev/null | grep -vq "Micro"; then
            last_step_seen=$now
            echo "last_step_seen_epoch $now" >> "$node_info"
        fi
        if [ -z "$last_step_seen" ]; then
            if { exited trainer_torchrun && [ "$(exit_rc trainer_torchrun)" != 0 ]; } \
                || { exited orchestrator && [ "$(exit_rc orchestrator)" != 0 ]; }; then
                verdict=CRASH
            elif [ -f "$abort_file" ]; then
                verdict=ABORT
            elif (( now - start_time > RUN_TIMEOUT_S )); then
                verdict=TIMEOUT
            fi
        elif (( now - last_step_seen > TEARDOWN_TIMEOUT_S )); then
            verdict=HANG
        fi
        [ -n "$verdict" ] && break
    done

    if [ -n "$verdict" ]; then
        touch "$abort_file"
        local stuck=() all=() pid
        exited orchestrator || stuck+=("$(comp_pid orchestrator)")
        for pid in $(descendants "$(comp_pid trainer_torchrun)"); do
            case "$(cat "/proc/$pid/comm" 2>/dev/null)" in
                bash | comp.sh | rank.sh) ;;
                *) stuck+=("$pid") ;;
            esac
        done
        for pid in "${stuck[@]}"; do
            all+=("$pid" $(descendants "$pid"))
        done
        {
            echo "verdict $verdict at $EPOCHREALTIME, stuck pids: ${stuck[*]}, with descendants: ${all[*]}"
            "$harness_dir/procdump.sh" "${all[@]}"
        } >> "$bench/hang_dump_train.txt" 2>&1
        kill -KILL "${all[@]}" 2>/dev/null
        local waited=0
        while ! { exited trainer_torchrun && exited orchestrator; }; do
            sleep 1
            waited=$((waited + 1))
            (( waited == 30 )) && kill -KILL "$(comp_pid trainer_torchrun)" 2>/dev/null
        done
    fi
    echo "train_node_finished_epoch $EPOCHREALTIME verdict ${verdict:-CLEAN} torchrun_rc $(exit_rc trainer_torchrun) orchestrator_rc $(exit_rc orchestrator)" >> "$node_info"
    touch "$train_finished_file"
}

run_infer_node() {
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
    export VLLM_ENGINE_READY_TIMEOUT_S="4200"
    export UCX_TLS="all"
    local local_ip
    local_ip=$(hostname -I | awk '{print $1}')
    ulimit -l unlimited

    PYTHONPYCACHEPREFIX=$cache_root/pycache "$harness_dir/comp.sh" env_pool "$bench/env_pool.log" \
        env-worker-pool @ "$CONFIG_DIR/envs/train/reverse-text.json" \
        --serve.address "tcp://$local_ip:0" \
        --address-file "$CONFIG_DIR/envs/train/reverse-text.0.address" &

    "$harness_dir/comp.sh" router "$bench/router.log" \
        vllm-router --policy sticky_least_loaded --host 0.0.0.0 --port 8000 \
        --request-id-headers x-session-id \
        --request-timeout-secs 14400 \
        --prometheus-port 29000 \
        --worker-startup-timeout-secs 4200 --log-level debug \
        --worker-urls "${router_workers[@]}" &

    local d port rpcbase
    for d in 0 1 2 3; do
        port=${backend_ports[$d]}
        rpcbase=/tmp/vllm-rpc-$USER-$PRL_RUN_NAME-$port
        mkdir -p "$rpcbase"
        CUDA_VISIBLE_DEVICES="$((2 * d)),$((2 * d + 1))" VLLM_RPC_BASE_PATH="$rpcbase" \
            "$harness_dir/comp.sh" "vllm_$d" "$bench/vllm_$d.log" \
            inference @ "$CONFIG_DIR/inference.json" \
            --server.host 0.0.0.0 --server.port "$port" \
            --vllm.data-parallel-size 1 --vllm.data-parallel-size-local 1 --vllm.api-server-count 1 &
    done

    local components=(vllm_0 vllm_1 vllm_2 vllm_3 router env_pool)
    local start_time=$EPOCHSECONDS verdict="" name
    while true; do
        sleep 2
        [ -f "$train_finished_file" ] && break
        for name in "${components[@]}"; do
            if exited "$name"; then
                verdict="EARLY_EXIT_$name"
            fi
        done
        if [ -z "$verdict" ] && [ -f "$abort_file" ]; then
            verdict=ABORT
        elif [ -z "$verdict" ] && (( EPOCHSECONDS - start_time > RUN_TIMEOUT_S + TEARDOWN_TIMEOUT_S + 300 )); then
            verdict=TIMEOUT
        fi
        if [ -n "$verdict" ]; then
            touch "$abort_file"
            while [ ! -f "$train_finished_file" ] && (( EPOCHSECONDS - start_time < RUN_TIMEOUT_S + 2 * TEARDOWN_TIMEOUT_S + 600 )); do
                sleep 2
            done
            break
        fi
    done

    local teardown=$bench/infer_teardown.txt
    for d in 0 1 2 3; do
        echo "health vllm_$d $(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://localhost:${backend_ports[$d]}/health")" >> "$teardown"
    done

    declare -A owner exit_after
    local pid main_pids=() tracked=()
    for name in "${components[@]}"; do
        pid=$(comp_pid "$name")
        [ -n "$pid" ] || continue
        main_pids+=("$pid")
        owner[$pid]="$name main $(cat "/proc/$pid/comm" 2>/dev/null)"
        tracked+=("$pid")
        for child in $(descendants "$pid"); do
            [ "$(cat "/proc/$child/comm" 2>/dev/null)" = comp.sh ] && continue
            owner[$child]="$name child $(cat "/proc/$child/comm" 2>/dev/null)"
            tracked+=("$child")
        done
    done
    local term_epoch=$EPOCHREALTIME
    echo "sigterm_epoch $term_epoch pre_verdict ${verdict:-NONE}" >> "$teardown"
    kill -TERM "${main_pids[@]}" 2>/dev/null

    local alive elapsed
    while true; do
        alive=()
        for pid in "${tracked[@]}"; do
            [ -n "${exit_after[$pid]:-}" ] && continue
            if [ -d "/proc/$pid" ] && [ "$(proc_state "$pid")" != Z ]; then
                alive+=("$pid")
            else
                exit_after[$pid]=$(awk -v a="$EPOCHREALTIME" -v b="$term_epoch" 'BEGIN { printf "%.1f", a - b }')
            fi
        done
        (( ${#alive[@]} == 0 )) && break
        elapsed=$(awk -v a="$EPOCHREALTIME" -v b="$term_epoch" 'BEGIN { printf "%d", a - b }')
        if (( elapsed > TEARDOWN_TIMEOUT_S )); then
            {
                echo "HANG after SIGTERM: ${alive[*]}"
                "$harness_dir/procdump.sh" "${alive[@]}"
            } >> "$bench/hang_dump_infer.txt" 2>&1
            kill -KILL "${alive[@]}" 2>/dev/null
            for pid in "${alive[@]}"; do
                exit_after[$pid]=HANG
            done
            break
        fi
        sleep 0.5
    done
    for pid in "${tracked[@]}"; do
        echo "proc $pid ${owner[$pid]} exit_after_sigterm_s ${exit_after[$pid]:-unknown}" >> "$teardown"
    done
    for name in "${components[@]}"; do
        while ! exited "$name"; do sleep 0.5; done
    done
    echo "infer_node_finished_epoch $EPOCHREALTIME verdict ${verdict:-CLEAN}" >> "$node_info"
}

if [ "$role" = train ]; then
    run_train_node
else
    run_infer_node
fi
rm -rf "$cache_root" /tmp/vllm-rpc-"$USER"-"$PRL_RUN_NAME"-*
exit 0
