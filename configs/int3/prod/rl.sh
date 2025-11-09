#!/bin/bash

#SBATCH --job-name=int3
#SBATCH --nodes=44
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive                 # Get the ENTIRE node exclusively
#SBATCH --output=/shared/logs/job_%j.log
#SBATCH --error=/shared/logs/job_%j.log

# Configs
export CKPT_STEP=${CKPT_STEP:-60}
export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"qwen3_coder"}

export NUM_TRAIN_NODES=${NUM_TRAIN_NODES:-16}
export NUM_INFER_NODES=${NUM_INFER_NODES:-28}

if [ $((NUM_TRAIN_NODES + NUM_INFER_NODES)) != $SLURM_JOB_NUM_NODES ]; then
    echo "NUM_TRAIN_NODES + NUM_INFER_NODES must equal SLURM_JOB_NUM_NODES"
    exit 1
fi

# Paths
export BASE_DIR=${BASE_DIR:-"/shared/prime-rl"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/shared/outputs/$SLURM_JOB_NAME"}
mkdir -p $OUTPUT_DIR/slurm

# Clear previous weights and rollouts
# WARN: This is potentially dangerous and could lead to data loss
# rm -rf $OUTPUT_DIR/weights $OUTPUT_DIR/rollouts
rm -rf $OUTPUT_DIR/rollouts

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Networking
export HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export TRAIN_HOSTS=${HOSTNAMES[@]:0:$NUM_TRAIN_NODES}
export INFER_HOSTS=${HOSTNAMES[@]:$NUM_TRAIN_NODES:$SLURM_JOB_NUM_NODES}
export BROADCAST_PORT=${BROADCAST_PORT:-29501}
export BROADCAST_TIMEOUT=${BROADCAST_TIMEOUT:-12000}
export NCCL_COMM_TIMEOUT=${NCCL_BROADCAST_TIMEOUT:-12000}

INFER_URLS=""
for host in ${INFER_HOSTS[@]}; do
    if [ -z "$INFER_URLS" ]; then
        INFER_URLS="http://$host:8000/v1"
    else
        INFER_URLS="$INFER_URLS,http://$host:8000/v1"
    fi
done
export INFER_URLS
echo "HOSTNAMES=${HOSTNAMES[@]}"
echo "TRAIN_HOSTS=${TRAIN_HOSTS[@]}"
echo "INFER_HOSTS=${INFER_HOSTS[@]}"
echo "INFER_URLS=${INFER_URLS}"

export MASTER_ADDR="${HOSTNAMES[0]}"
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# Cleanup
srun bash -c 'pkill -9 -f torchrun || true && fuser -k ${MASTER_PORT}/tcp || true'
srun bash -c 'pkill -9 -f prime-rl || true'
srun bash -c 'pkill -9 -f VLLM || true && fuser -k 8000/tcp || true'

# Install environment
cd $BASE_DIR
source .env
source .venv/bin/activate

# Install environment as local package
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Run RL
srun bash -c '
    # Setup environment
    cd $BASE_DIR
    source .env
    source .venv/bin/activate

    # Disable sync to avoid conflicts with lockfile

    # Higher ulimit
    ulimit -n 65536
    export GIT_LFS_SKIP_SMUDGE=1

    # Infiniband setup
    IB_HCA=$(ibv_devinfo | sed -n -e '/hca_id/p' -e '/link_layer:/p' | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' | tr -d '[[:blank:]]' |paste -sd,)
    export NCCL_IB_HCA=$IB_HCA

    if [ $SLURM_PROCID -ge $NUM_TRAIN_NODES ]; then
        # This is required for vLLM graph compile to work
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
        INFER_NODE_RANK=$((SLURM_PROCID - NUM_TRAIN_NODES))
        uv run inference \
        @ configs/int3/prod/infer.toml \
        --weight_broadcast.type nccl \
        --enable-auto-tool-choice \
        --tool-call-parser $TOOL_CALL_PARSER \
        --enable-log-requests \
        --no-disable-log-requests \
        2>&1 | tee $OUTPUT_DIR/slurm/latest_infer_node_rank_${INFER_NODE_RANK}.log $OUTPUT_DIR/slurm/job_${INFER_NODE_RANK}_infer_node_rank_${INFER_NODE_RANK}.log
    else

        if [ "$SLURM_PROCID" -eq 0 ]; then
            uv run orchestrator \
            @ configs/int3/prod/orch.toml \
            --ckpt.resume_step $CKPT_STEP \
            --weight_broadcast.type nccl \
            --weight_broadcast.host $MASTER_ADDR \
            --weight_broadcast.port $BROADCAST_PORT \
            --weight_broadcast.timeout $BROADCAST_TIMEOUT \
            --client.base-url $INFER_URLS \
            --client.timeout 3600 \
            --num-train-workers $((NUM_TRAIN_NODES * 8)) \
            --output-dir $OUTPUT_DIR \
            2>&1 | tee $OUTPUT_DIR/slurm/latest_orchestrator.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_orchestrator.log & disown
        fi

        export HF_HUB_OFFLINE=1


        # This is required for compilation to work correctly
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" 

        echo $HOSTNAMES | tee  $OUTPUT_DIR/slurm/latest_train_node_rank_${SLURM_PROCID}.log
        uv run torchrun \
        --nnodes=$NUM_TRAIN_NODES \
        --nproc-per-node=8 \
        --node-rank=$SLURM_PROCID \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv-id=job_$SLURM_JOB_ID \
        --log-dir=$OUTPUT_DIR/torchrun \
        --tee=3 \
        --redirects=3 \
        --local-ranks-filter=0 \
        src/prime_rl/trainer/rl/train.py \
        @ configs/int3/prod/train.toml \
        --weight_broadcast.type nccl \
        --weight_broadcast.host 0.0.0.0 \
        --weight_broadcast.port $BROADCAST_PORT \
        --weight_broadcast.inference_world_size $((8 * NUM_INFER_NODES))\
        --weight_broadcast.timeout $BROADCAST_TIMEOUT \
        --dist_timeout_seconds $NCCL_COMM_TIMEOUT \
        --model.dp-replicate $((NUM_TRAIN_NODES / 8)) \
        --ckpt.resume_step $CKPT_STEP \
        --output-dir $OUTPUT_DIR \
        2>&1 | tee -a $OUTPUT_DIR/slurm/latest_train_node_rank_${SLURM_PROCID}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_train_node_rank_${SLURM_PROCID}.log
    fi
'
