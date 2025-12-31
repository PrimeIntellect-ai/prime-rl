set -e

export BS=${BS:-8}
export SEQ=${SEQ:-8192}
export WORLD_SIZE=${WORLD_SIZE:-8}
export RUN_NAME=${RUN_NAME:-debug-sft}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-${RUN_NAME}}

OUTPUT_DIR="/data/prime-rl/outputs/${RUN_NAME}"
export OUTPUT_DIR

mkdir -p $OUTPUT_DIR/logs

ROOT_DIR=$(pwd)

echo "ROOT_DIR: $ROOT_DIR"

bash -c '
    cd "$ROOT_DIR"
    source /home/ubuntu/workspace/prime-rl/.env

    source .venv/bin/activate

    ulimit -n 32000

    uv run torchrun \
        --nproc-per-node ${WORLD_SIZE} \
        --local-ranks-filter 0 \
        src/prime_rl/trainer/rl/train.py \
        --output-dir $OUTPUT_DIR \
        --model.name Qwen/Qwen3-30B-A3B \
        --model.impl custom \
        --model.debug.num_layers 2 \
        --model.debug.random_init \
        --model.seq_len ${SEQ} \
        --data.fake.batch_size ${BS} \
        --bench \
        --log.level debug \
        --wandb.project chunked-loss \
        --wandb.name $WANDB_RUN_NAME \
        2>&1 | tee $OUTPUT_DIR/logs/${RUN_NAME}.out
'
