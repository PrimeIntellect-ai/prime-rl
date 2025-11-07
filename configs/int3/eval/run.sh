#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=cluster
#SBATCH --exclusive                 # Get the ENTIRE node exclusively
#SBATCH --output=/shared/logs/job_%j.log
#SBATCH --error=/shared/logs/job_%j.log


# Configs
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export EVAL_MODE=${EVAL_MODE:-"debug"}
export EVAL_CONFIG=${EVAL_CONFIG:-"configs/int3/eval/debug-eval.toml"}
export EXTRAS=${EXTRAS:-""}

if [[ "$MODEL_NAME" == "PrimeIntellect/INTELLECT-3"* ]]; then
    export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"qwen3_coder"}
elif [[ "$MODEL_NAME" == "zai-org/GLM-4.5-Air"* ]]; then
    export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"glm45"}
elif [[ "$MODEL_NAME" == "Qwen/Qwen3"* ]]; then
    export TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"qwen3_coder"}
else
    echo "Invalid model name: $MODEL_NAME. Only support GLM-4.5-Air, INTELLECT-3, and Qwen3 in this script for now."
    exit 1
fi

# Update job name to include model slug
export MODEL_ORG=$(echo "$MODEL_NAME" | cut -d'/' -f1)
export MODEL_ID=$(echo "$MODEL_NAME" | cut -d'/' -f2)
export JOB_NAME="$(echo $MODEL_ID | tr '[:upper:]' '[:lower:]')-${EVAL_MODE}-evals"
scontrol update JobId=$SLURM_JOB_ID JobName="${JOB_NAME}" 2>/dev/null || true

# Paths
export BASE_DIR=${BASE_DIR:-"/shared/prime-rl"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/shared/outputs/evals"}
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
srun bash -c 'pkill -9 -f VLLM || true && fuser -k 8000/tcp || true'

# Install environment on head node
cd $BASE_DIR
unset $VIRTUAL_ENV
source .env
source .venv/bin/activate

# Predownload the model
uv tool install huggingface-hub[cli]
hf download $MODEL_NAME

# Install environments (check if already installed first)
uv tool install prime
envs_to_install=(
    "primeintellect/math500"
    "primeintellect/aime2024" 
    "primeintellect/aime2025"
    "primeintellect/gpqa"
    "primeintellect/simpleqa-verified"
    "primeintellect/mmlu-pro"
    "primeintellect/hle"
    "primeintellect/livecodebench"
    "primeintellect/ifeval"
)

for env in "${envs_to_install[@]}"; do
    env_org="${env | cut -d'/' -f1}"
    env_name="${env | cut -d'/' -f2}"
    
    if ! uv run python -c "import ${env_name}" 2>/dev/null; then
        echo "Installing ${env}..."
        prime env install ${env}
    else
        echo "${env} already installed, skipping..."
    fi
done

# Handle tau2-bench separately since it uses 'prime env pull'
if ! uv run python -c "import tau2_bench" 2>/dev/null; then
    echo "Installing primeintellect/tau2-bench..."
    prime env pull primeintellect/tau2-bench
    uv pip install -e /shared/prime-rl/primeintellect-tau2-bench-latest
else
    echo "primeintellect/tau2-bench already installed, skipping..."
fi

# Run inference and evaluation
srun bash -c '
    # Setup environment
    cd $BASE_DIR
    source .env
    source .venv/bin/activate

    # Higher ulimit
    ulimit -n 65535

    if [ "$SLURM_PROCID" -eq 0 ]; then
        # Run evaluation in the background
        (
            uv run eval \
            --client.base-url $INFER_URLS \
            --model.name $MODEL_NAME \
            @ $EVAL_CONFIG \
            --wandb.project intellect-3-evals \
            --wandb.name ${JOB_NAME} \
            --save.hf.dataset-name PrimeIntellect/${MODEL_ID}-${EVAL_MODE^}-Evals \
            --save.hf.private \
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
    uv run inference \
    --model.name $MODEL_NAME \
    --enable-auto-tool-choice \
    --tool-call-parser $TOOL_CALL_PARSER \
    --tensor-parallel-size 8 \
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
