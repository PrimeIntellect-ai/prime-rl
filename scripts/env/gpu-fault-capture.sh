#!/usr/bin/env bash
# Per-node GPU fault capture. We are otherwise blind to HW-vs-software when an
# engine dies (RUN-2: vLLM TP rank hit a CUDA "illegal instruction" mid-decode;
# decisive Xid/dmesg evidence was lost when the allocation released). This
# snapshots GPU health into the run's log dir so the NEXT fault is diagnosable.
#
# Unprivileged sources (validated on Isambard GH200, driver 565.57.01):
#   - `nvidia-smi -q`            full dump: ECC errors, retired pages, remapped
#                                rows, throttle reasons (the downstream symptoms
#                                of a HW fault).
#   - `nvidia-smi --query-gpu`   compact one-line-per-GPU health row.
#   - `dmesg`                    authoritative Xid source, but kernel.dmesg_restrict=1
#                                blocks unprivileged reads on this cluster — we
#                                detect that and record UNAVAILABLE (never swallow).
# `dcgmi` is not installed here, so it is not used.
#
# Two capture points: a periodic background snapshot and an on-exit snapshot
# installed via an EXIT trap on the calling shell. Source AFTER any other EXIT
# trap is set by the caller — we chain, we do not clobber.
#
# LIMITATION: a SIGKILL teardown (srun --kill-on-bad-exit, scancel, OOM-killer)
# kills the shell without running the EXIT trap, so the on-exit snapshot will NOT
# fire in that case. The periodic background snapshot is then the only guaranteed
# signal — keep GPU_FAULT_SNAPSHOT_INTERVAL_SECONDS short enough to bracket a fault.
#
# Usage (from a launch script that has $OUTPUT_DIR set):
#   GPU_FAULT_LOG_DIR="$OUTPUT_DIR/logs/inference" source .../gpu-fault-capture.sh
# Defaults to $OUTPUT_DIR/logs/inference if GPU_FAULT_LOG_DIR is unset.

_gpu_fault_log_dir="${GPU_FAULT_LOG_DIR:-${OUTPUT_DIR:?GPU_FAULT_LOG_DIR or OUTPUT_DIR required}/logs/inference}"
mkdir -p "$_gpu_fault_log_dir"
_gpu_fault_host="$(hostname -s)"
_gpu_fault_file="$_gpu_fault_log_dir/gpu_health_${_gpu_fault_host}.log"
_gpu_fault_interval="${GPU_FAULT_SNAPSHOT_INTERVAL_SECONDS:-300}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-fault-capture] FATAL: nvidia-smi not found on $_gpu_fault_host; cannot capture GPU health" >&2
    exit 1
fi

# One health snapshot appended to the per-host log. $1 is the trigger label.
gpu_fault_snapshot() {
    local trigger="$1"
    local ts
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    {
        echo "===== [gpu-fault-capture] host=$_gpu_fault_host trigger=$trigger ts=$ts ====="
        echo "--- nvidia-smi --query-gpu (compact health) ---"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,clocks_event_reasons.active,ecc.errors.uncorrected.aggregate.total,ecc.errors.uncorrected.volatile.total,retired_pages.pending,remapped_rows.pending,remapped_rows.uncorrectable \
            --format=csv 2>&1
        echo "--- nvidia-smi -q (ECC / retired pages / remapped rows / throttle) ---"
        nvidia-smi -q 2>&1
        echo "--- dmesg (kernel Xid) ---"
        if dmesg 2>/dev/null | tail -n 200; then
            :
        else
            echo "[gpu-fault-capture] dmesg UNAVAILABLE (kernel.dmesg_restrict=$(cat /proc/sys/kernel/dmesg_restrict 2>/dev/null || echo '?'), unprivileged read blocked) — Xid not captured from this node"
        fi
        echo "===== [gpu-fault-capture] end trigger=$trigger ====="
    } >> "$_gpu_fault_file" 2>&1
}

# Periodic background snapshotter. Self-terminates with the calling shell because
# it is a child job; teardown loops in the launch templates kill leftover jobs.
gpu_fault_snapshot_loop() {
    while true; do
        sleep "$_gpu_fault_interval"
        gpu_fault_snapshot periodic
    done
}

echo "[gpu-fault-capture] host=$_gpu_fault_host writing GPU health to $_gpu_fault_file (periodic=${_gpu_fault_interval}s, on-exit trap)"
gpu_fault_snapshot startup
gpu_fault_snapshot_loop &
GPU_FAULT_LOOP_PID="$!"

# Chain onto any existing EXIT trap so we snapshot on engine/process exit without
# clobbering the caller's teardown.
_gpu_fault_prev_exit_trap="$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/")"
trap "gpu_fault_snapshot exit; kill \"\$GPU_FAULT_LOOP_PID\" 2>/dev/null || true; ${_gpu_fault_prev_exit_trap}" EXIT
