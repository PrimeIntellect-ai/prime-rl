#!/bin/bash

SESSION="rl_run"

# Define commands as blocks
read -r -d '' INFERECE_CMD << 'EOF'
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run python src/zeroband/infer.py @ configs/inference/simple_reverse_two_gpus.toml
EOF

read -r -d '' TRAINING_CMD << 'EOF'
ulimit -n 4096
export CUDA_VISIBLE_DEVICES=1
uv run torchrun --nproc_per_node=1 src/zeroband/train.py @ configs/training/simple_reverse_two_gpu.toml
EOF

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new session and split
tmux new-session -d -s $SESSION
tmux split-window -h -t $SESSION

# Send command blocks to each pane
tmux send-keys -t $SESSION:0.0 "$INFERECE_CMD" Enter
tmux send-keys -t $SESSION:0.1 "$TRAINING_CMD" Enter

# Attach to session
tmux attach -t $SESSION 