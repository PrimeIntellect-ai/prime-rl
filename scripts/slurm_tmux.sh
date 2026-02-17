#!/bin/bash

JOB_NAME="$1"
OUTPUT_DIR="$2"

if [[ -z "$JOB_NAME" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 JOB_NAME OUTPUT_DIR" >&2
  exit 1
fi

SLURM_LOG_DIR="${OUTPUT_DIR}/slurm"
SESSION_NAME="slurm-${JOB_NAME}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Attaching to existing tmux session: $SESSION_NAME"
  exec tmux attach-session -t "$SESSION_NAME"
fi

echo "Creating tmux session: $SESSION_NAME"

# Window 0: Terminal
tmux new-session -d -s "$SESSION_NAME" -n "Terminal"

# Window 1: Logs - 3 vertical panes
tmux new-window -t "$SESSION_NAME" -n "Logs"

tmux split-window -v -t "$SESSION_NAME:Logs.0"
tmux split-window -v -t "$SESSION_NAME:Logs.1"
tmux select-layout -t "$SESSION_NAME:Logs" even-vertical

tmux select-pane -t "$SESSION_NAME:Logs.0" -T "Trainer"
tmux select-pane -t "$SESSION_NAME:Logs.1" -T "Orchestrator"
tmux select-pane -t "$SESSION_NAME:Logs.2" -T "Inference"

tmux send-keys -t "$SESSION_NAME:Logs.0" \
  "tail -F ${SLURM_LOG_DIR}/latest_train_node_rank_*.log 2>/dev/null" C-m

tmux send-keys -t "$SESSION_NAME:Logs.1" \
  "tail -F ${SLURM_LOG_DIR}/latest_orchestrator.log 2>/dev/null" C-m

tmux send-keys -t "$SESSION_NAME:Logs.2" \
  "tail -F ${SLURM_LOG_DIR}/latest_infer_node_rank_*.log 2>/dev/null" C-m

# Pane title styling
tmux set-option -t "$SESSION_NAME" -g pane-border-status top
tmux set-option -t "$SESSION_NAME" -g pane-border-format " #{pane_title} "

# Focus Terminal window and attach
tmux select-window -t "$SESSION_NAME:Terminal"
exec tmux attach-session -t "$SESSION_NAME"
