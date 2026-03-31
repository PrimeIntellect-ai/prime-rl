#!/bin/bash

SESSION_NAME="prime-rl"
OUTPUT_DIR="outputs"

# Optional CLI parsing
# Supports:
#   -s|--session-name NAME
#   -o|--output-dir DIR
#   Positional: [SESSION_NAME [OUTPUT_DIR]]
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--session-name)
      if [[ -z "$2" ]]; then
        echo "Error: --session-name requires a value" >&2
        exit 1
      fi
      SESSION_NAME="$2"
      shift 2
      ;;
    -o|--output-dir)
      if [[ -z "$2" ]]; then
        echo "Error: --output-dir requires a value" >&2
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-s SESSION_NAME] [-o OUTPUT_DIR] [SESSION_NAME [OUTPUT_DIR]]" >&2
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  SESSION_NAME="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
  OUTPUT_DIR="${POSITIONAL[1]}"
fi

LOG_DIR="${OUTPUT_DIR}/logs"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Attaching to tmux session: $SESSION_NAME"
  exec tmux attach-session -t "$SESSION_NAME"
fi

echo "Creating new tmux session: $SESSION_NAME"

# Window 1: Logs - 4 vertical panes
tmux new-session -d -s "$SESSION_NAME" -n "Logs"

tmux split-window -v -t "$SESSION_NAME:Logs.0"
tmux split-window -v -t "$SESSION_NAME:Logs.1"
tmux split-window -v -t "$SESSION_NAME:Logs.2"
tmux select-layout -t "$SESSION_NAME:Logs" even-vertical

tmux select-pane -t "$SESSION_NAME:Logs.0" -T "Trainer"
tmux select-pane -t "$SESSION_NAME:Logs.1" -T "Orchestrator"
tmux select-pane -t "$SESSION_NAME:Logs.2" -T "Envs"
tmux select-pane -t "$SESSION_NAME:Logs.3" -T "Inference"

tmux send-keys -t "$SESSION_NAME:Logs.0" \
  "tail -F ${LOG_DIR}/trainer.log 2>/dev/null" C-m

tmux send-keys -t "$SESSION_NAME:Logs.1" \
  "tail -F ${LOG_DIR}/orchestrator.log 2>/dev/null" C-m

tmux send-keys -t "$SESSION_NAME:Logs.2" \
  "tail -F ${LOG_DIR}/envs/*/*/*.log 2>/dev/null" C-m

tmux send-keys -t "$SESSION_NAME:Logs.3" \
  "tail -F ${LOG_DIR}/inference.log 2>/dev/null" C-m

# Window 2: Monitor
tmux new-window -t "$SESSION_NAME" -n "Monitor"
tmux split-window -h -t "$SESSION_NAME:Monitor"
tmux select-layout -t "$SESSION_NAME:Monitor" even-horizontal

tmux select-pane -t "$SESSION_NAME:Monitor.0" -T "GPU"
tmux send-keys -t "$SESSION_NAME:Monitor.0" "nvtop" C-m

tmux select-pane -t "$SESSION_NAME:Monitor.1" -T "CPU"
tmux send-keys -t "$SESSION_NAME:Monitor.1" "htop" C-m

# Pane title styling
tmux set-option -t "$SESSION_NAME" -g pane-border-status top
tmux set-option -t "$SESSION_NAME" -g pane-border-format " #{pane_title} "

# Focus logs window and attach
tmux select-window -t "$SESSION_NAME:Logs"
tmux select-pane -t "$SESSION_NAME:Logs.0"
exec tmux attach-session -t "$SESSION_NAME"
