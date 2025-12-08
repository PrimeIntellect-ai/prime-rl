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

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Attaching to tmux session: $SESSION_NAME"
  exec tmux attach-session -t "$SESSION_NAME"
else
  echo "Creating new tmux session: $SESSION_NAME"

  # Start new tmux session with first window
  tmux new-session -d -s "$SESSION_NAME" -n "RL"

  # Window 1: RL - 3 vertical panes
  tmux split-window -v -t "$SESSION_NAME:RL.0"
  tmux split-window -v -t "$SESSION_NAME:RL.1"
  tmux select-layout -t "$SESSION_NAME:RL" even-vertical

  # Pane titles
  tmux select-pane -t "$SESSION_NAME:RL.0" -T "Trainer"
  tmux select-pane -t "$SESSION_NAME:RL.1" -T "Orchestrator"
  tmux select-pane -t "$SESSION_NAME:RL.2" -T "Inference"

  # Logs: Orchestrator
  tmux send-keys -t "$SESSION_NAME:RL.1" 'while true; do
echo "Waiting for orchestrator log file..."
while [ ! -f '"$OUTPUT_DIR"'/logs/orchestrator.stdout ]; do sleep 1; done
echo "Following orchestrator.stdout..."
tail -F '"$OUTPUT_DIR"'/logs/orchestrator.stdout
done' C-m

  # Logs: Inference
  tmux send-keys -t "$SESSION_NAME:RL.2" 'while true; do
echo "Waiting for inference log file..."
while [ ! -f '"$OUTPUT_DIR"'/logs/inference.stdout ]; do sleep 1; done
echo "Following inference.stdout..."
tail -F '"$OUTPUT_DIR"'/logs/inference.stdout
done' C-m

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
  tmux set-window-option -t "$SESSION_NAME:RL" pane-border-status top
  tmux set-window-option -t "$SESSION_NAME:Monitor" pane-border-status top

  # Focus trainer pane and attach
  tmux select-window -t "$SESSION_NAME:RL"
  tmux select-pane -t "$SESSION_NAME:RL.0"
  exec tmux attach-session -t "$SESSION_NAME"
fi