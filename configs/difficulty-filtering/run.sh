#!/bin/bash

#SBATCH --job-name=difficulty-filtering
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive                 # Get the ENTIRE node exclusively
#SBATCH --output=/shared/logs/job_%j.log
#SBATCH --error=/shared/logs/job_%j.log

# Configs
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-4B-Instruct-2507"}
export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"hermes"}
export EVAL_CONFIG=${EVAL_CONFIG:-"configs/difficulty-filtering/math.toml"}
export DP=${DP:-8}
export TP=${TP:-1}
export EXTRAS=${EXTRAS:-""}

# Update job name to include model slug
export MODEL_ORG=$(echo "$MODEL_NAME" | cut -d'/' -f1)
export MODEL_ID=$(echo "$MODEL_NAME" | cut -d'/' -f2)
export JOB_NAME="$(echo $MODEL_ID | tr '[:upper:]' '[:lower:]')-${EVAL_MODE}-evals"
scontrol update JobId=$SLURM_JOB_ID JobName="${JOB_NAME}" 2>/dev/null || true

# Paths
export BASE_DIR=${BASE_DIR:-"/home/mika/prime-rl"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/home/mika/outputs"}
mkdir -p $OUTPUT_DIR/slurm

# General
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Networking
export HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )

# Build inference URLs
INFER_URLS=""
for host in ${HOSTNAMES[@]}; do
    if [ -z "$INFER_URLS" ]; then
        INFER_URLS="http://$host:8000/v1"
    else
        INFER_URLS="$INFER_URLS,http://$host:8000/v1"
    fi
done
export INFER_URLS

echo "HOSTNAMES=${HOSTNAMES[@]}"
echo "INFER_URLS=${INFER_URLS}"

# Cleanup
srun bash -c 'sudo pkill -9 -f VLLM || sudo pkill -f torchrun || true && sudo pkill -f prime-rl || true && sudo fuser -k 8000/tcp || true'

# Install environment on head node
cd $BASE_DIR
source ~/.env
source $BASE_DIR/.venv/bin/activate

# Run inference and evaluation
srun bash -c '
    # Setup environment
    cd $BASE_DIR
    source ~/.env
    source $BASE_DIR/.venv/bin/activate

    # Higher ulimit
    ulimit -n 65536

    if [ "$SLURM_PROCID" -eq 0 ]; then
        # Run evaluation in the background
        (
            uv run --no-sync eval \
            --client.base-url $INFER_URLS \
            @ $EVAL_CONFIG \
            --output-dir $OUTPUT_DIR \
            2>&1 | tee $OUTPUT_DIR/slurm/latest_eval.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_eval.log
            
            # Create a completion marker file when evaluation finishes
            touch $OUTPUT_DIR/slurm/eval_complete_${SLURM_JOB_ID}
        ) & disown
        EVAL_PID=$!
        echo "Started evaluation in background with PID: $EVAL_PID"
    fi

    # Infiniband setup
    IB_HCA=$(ibv_devinfo | sed -n -e '/hca_id/p' -e '/link_layer:/p' | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' | tr -d '[[:blank:]]' |paste -sd,)
    export NCCL_IB_HCA=$IB_HCA

    # This is required for vLLM graph compile to work
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
    
    # Start inference server in background
    uv run vllm serve $MODEL_NAME \
    --enable-auto-tool-choice \
    --tool-call-parser $TOOL_CALL_PARSER \
    --tensor-parallel-size $TP \
    --data-parallel-size $DP \
    $EXTRAS \
    2>&1 | tee -a $OUTPUT_DIR/slurm/latest_infer_node_rank_${SLURM_PROCID}.log $OUTPUT_DIR/slurm/job_${SLURM_JOB_ID}_infer_node_rank_${SLURM_PROCID}.log & disown
    
    VLLM_PID=$!
    echo "Started vLLM server with PID: $VLLM_PID on rank ${SLURM_PROCID}"
    
    # Wait for evaluation to complete (only on rank 0, others just wait for termination signal)
    if [ "$SLURM_PROCID" -eq 0 ]; then
        echo "Waiting for evaluation to complete..."
        while [ ! -f "$OUTPUT_DIR/slurm/eval_complete_${SLURM_JOB_ID}" ]; do
            sleep 10
            echo "Still waiting for evaluation to complete..."
        done
        echo "Evaluation completed! Signaling all nodes to terminate..."
        
        # Create termination signal for all nodes
        touch $OUTPUT_DIR/slurm/terminate_all_${SLURM_JOB_ID}
    else
        # Non-rank-0 processes wait for termination signal
        while [ ! -f "$OUTPUT_DIR/slurm/terminate_all_${SLURM_JOB_ID}" ]; do
            sleep 10
        done
        echo "Received termination signal on rank ${SLURM_PROCID}"
    fi
    
    # Kill the vLLM server
    echo "Terminating vLLM server (PID: $VLLM_PID) on rank ${SLURM_PROCID}"
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    
    echo "Node rank ${SLURM_PROCID} finished successfully"
'