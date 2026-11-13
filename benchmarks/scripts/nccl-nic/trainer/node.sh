#!/bin/bash
# Usage: node.sh (one task per node under srun; reads ARM_DIR, RUN_DIR, TRAIN_CONFIG_PATH, MASTER_ADDR, MASTER_PORT, MAX_STEPS, RUN_TIMEOUT_S, TEARDOWN_TIMEOUT_S)
# Mirrors the trainer branch of multi_node_sft.sbatch.j2 with a per-rank exit recorder and a teardown-hang watchdog.
set -uo pipefail

node_rank=1
[ "$(hostname)" = "$MASTER_ADDR" ] && node_rank=0
harness_dir=$(dirname "$(readlink -f "$0")")
node_dir=$RUN_DIR/bench/node_${node_rank}
export RANK_LOG_DIR=$RUN_DIR/bench/ranks
mkdir -p "$node_dir" "$RANK_LOG_DIR"

cd "$ARM_DIR"
source .venv/bin/activate
ulimit -n 65536

IB_HCA=$(ibv_devinfo 2>/dev/null | sed -n -e "/hca_id/p" -e "/link_layer:/p" | grep -B1 InfiniBand | grep hca_id | sed -e "s/^hca_id://g" | tr -d "[[:blank:]]" | paste -sd,)
[ -n "$IB_HCA" ] && export NCCL_IB_HCA=$IB_HCA

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="1"
export GIT_LFS_SKIP_SMUDGE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_FR_BUFFER_SIZE="2000"
export TORCH_NCCL_DUMP_ON_TIMEOUT="1"
export TORCH_FR_DUMP_TEMP_FILE="/tmp/nccl_trace_$(id -u)_rank_"
export NCCL_DEBUG=VERSION

cache_root=/tmp/nccl-nic-trainer-$(basename "$RUN_DIR")
rm -rf "$cache_root"
export TRITON_CACHE_DIR=$cache_root/triton
export TORCHINDUCTOR_CACHE_DIR=$cache_root/inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

{
    echo "host $(hostname) node_rank $node_rank arm $ARM_DIR commit $(git -C "$ARM_DIR" rev-parse --short=9 HEAD)"
    echo "NCCL_IB_HCA=${NCCL_IB_HCA:-} TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
    echo "gpu_mem_used_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd,)"
} > "$node_dir/node_info.txt"

torchrun \
    --role=trainer \
    --nnodes=2 \
    --nproc-per-node=8 \
    --node-rank="$node_rank" \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv-id="$(basename "$RUN_DIR")" \
    --no-python \
    "$harness_dir/rank.sh" \
    > "$node_dir/torchrun.log" 2>&1 &
torchrun_pid=$!

rank0_log=$RANK_LOG_DIR/rank_0.log
start_time=$EPOCHSECONDS
last_step_seen=""
verdict=""
while kill -0 "$torchrun_pid" 2>/dev/null; do
    sleep 2
    now=$EPOCHSECONDS
    if [ -z "$last_step_seen" ] && grep -q "Step $MAX_STEPS |" "$rank0_log" 2>/dev/null; then
        last_step_seen=$now
        echo "last_step_seen_epoch $now" >> "$node_dir/node_info.txt"
    fi
    if [ -n "$last_step_seen" ] && (( now - last_step_seen > TEARDOWN_TIMEOUT_S )); then
        verdict=HANG
    elif (( now - start_time > RUN_TIMEOUT_S )); then
        verdict=TIMEOUT
    fi
    if [ -n "$verdict" ]; then
        for pid in $(pgrep -u "$(id -u)" -f "prime_rl.trainer.sft.train"); do
            {
                echo "== pid $pid"
                cat "/proc/$pid/status" 2>/dev/null | grep -E "^(Name|State|Threads):"
                for task in /proc/$pid/task/*; do
                    echo "$(basename "$task") $(cat "$task/comm" 2>/dev/null) wchan=$(cat "$task/wchan" 2>/dev/null) state=$(awk '{print $3}' "$task/stat" 2>/dev/null)"
                done
                timeout 60 py-spy dump --pid "$pid" 2>&1 || true
            } >> "$node_dir/hang_dump.txt"
        done
        pkill -KILL -P "$torchrun_pid"
        pkill -KILL -u "$(id -u)" -f "prime_rl.trainer.sft.train"
        kill -KILL "$torchrun_pid" 2>/dev/null
        break
    fi
done
wait "$torchrun_pid"
rc=$?
echo "torchrun_exit_epoch $EPOCHREALTIME rc $rc verdict ${verdict:-CLEAN}" >> "$node_dir/node_info.txt"
rm -rf "$cache_root"
exit "$rc"
