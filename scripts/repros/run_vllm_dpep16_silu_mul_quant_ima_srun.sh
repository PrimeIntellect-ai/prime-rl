#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(git rev-parse --show-toplevel)}
NODES=${NODES:-2}
TASKS_PER_NODE=${TASKS_PER_NODE:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
PARTITION=${PARTITION:-cluster}
TIME_LIMIT=${TIME_LIMIT:-00:20:00}
MASTER_PORT=${MASTER_PORT:-29577}
SHAPE=${SHAPE:-dpep16-warmup}
APPLY_INT64_PATCH=${APPLY_INT64_PATCH:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/beegfs/outputs/vllm_dpep16_silu_ima_repro/$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUTPUT_DIR"

echo "output_dir=$OUTPUT_DIR"
echo "shape=$SHAPE apply_int64_patch=$APPLY_INT64_PATCH nodes=$NODES tasks_per_node=$TASKS_PER_NODE"

extra_srun_args=()
if [ -n "${NODELIST:-}" ]; then
    extra_srun_args+=(--nodelist="$NODELIST")
fi
if [ -n "${EXCLUDE:-}" ]; then
    extra_srun_args+=(--exclude="$EXCLUDE")
fi
if [ -n "${ACCOUNT:-}" ]; then
    extra_srun_args+=(--account="$ACCOUNT")
fi
if [ "${EXCLUSIVE:-1}" = "1" ]; then
    extra_srun_args+=(--exclusive)
fi

srun \
    --nodes="$NODES" \
    --ntasks-per-node="$TASKS_PER_NODE" \
    --gres="gpu:$GPUS_PER_NODE" \
    --cpus-per-task=8 \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    "${extra_srun_args[@]}" \
    bash -lc '
        set -euo pipefail
        cd '"$PROJECT_DIR"'

        export RANK=$SLURM_PROCID
        export WORLD_SIZE=$SLURM_NTASKS
        export LOCAL_RANK=$SLURM_LOCALID
        export MASTER_PORT='"$MASTER_PORT"'
        export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_LAUNCH_BLOCKING=1
        export VLLM_USE_DEEP_GEMM=1
        export VLLM_MOE_USE_DEEP_GEMM=1
        export PYTHONUNBUFFERED=1

        source .venv/bin/activate

        args=(--shape '"$SHAPE"')
        if [ '"$APPLY_INT64_PATCH"' = "1" ]; then
            args+=(--apply-fix)
        fi

        echo "[srun-rank $SLURM_PROCID] node=$SLURM_NODEID host=$(hostname) local_rank=$SLURM_LOCALID master=$MASTER_ADDR:$MASTER_PORT"
        uv run --extra disagg python scripts/repros/vllm_dpep16_silu_mul_quant_ima.py "${args[@]}"
    ' 2>&1 | tee "$OUTPUT_DIR/srun.log"
