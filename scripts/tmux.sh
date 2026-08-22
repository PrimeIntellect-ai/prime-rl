#!/bin/bash

SESSION_NAME="prime-rl"
OUTPUT_DIR="outputs"
AGENT="claude"

# Optional CLI parsing
# Supports:
#   -s|--session-name NAME
#   -o|--output-dir DIR     (directory grouping runs, or a specific run directory)
#   -a|--agent AGENT        (claude|codex, default: claude)
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
    -a|--agent)
      if [[ -z "$2" ]]; then
        echo "Error: --agent requires a value (claude|codex)" >&2
        exit 1
      fi
      AGENT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-s SESSION_NAME] [-o OUTPUT_DIR] [-a AGENT] [SESSION_NAME [OUTPUT_DIR]]" >&2
      echo "  -o, --output-dir  directory grouping runs, or a specific run directory (default: outputs)" >&2
      echo "  -a, --agent       claude|codex  (default: claude)" >&2
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

# Each run writes its logs to <output_dir>/<run_name>/logs/latest, and run names
# are auto-generated unless set via --run.name — so the run directory is unknown
# when the session starts. LOG_DIR_GLOB matches both an output directory and a
# run directory passed as OUTPUT_DIR; each pane polls it and tails the newest match.
LOG_DIR_GLOB="${OUTPUT_DIR}/logs/latest ${OUTPUT_DIR}/*/logs/latest"

# Build a pane command that waits for a run to appear, then tails PATTERN
# (relative to the newest logs/latest), optionally filtered through grep.
# ``command ls`` bypasses interactive aliases (e.g. ls=exa) in the pane shell.
tail_cmd() {
  local pattern="$1" filter="${2:-}"
  local tail="tail -F \$files"
  if [[ -n "$filter" ]]; then
    tail+=" | grep --line-buffered $filter"
  fi
  echo "while true; do dir=\$(command ls -td ${LOG_DIR_GLOB} 2>/dev/null | head -1); files=\$(command ls \$dir/$pattern 2>/dev/null); [ -n \"\$files\" ] && $tail; sleep 1; done"
}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Attaching to tmux session: $SESSION_NAME"
  exec tmux attach-session -t "$SESSION_NAME"
fi

echo "Creating new tmux session: $SESSION_NAME"

# Window 0: Launcher - empty shell
tmux new-session -d -s "$SESSION_NAME" -n "Launcher"

# Window 1: Logs - 4 vertical panes
tmux new-window -t "$SESSION_NAME" -n "Logs"

tmux split-window -v -t "$SESSION_NAME:Logs.0"
tmux split-window -v -t "$SESSION_NAME:Logs.1"
tmux split-window -v -t "$SESSION_NAME:Logs.2"
tmux select-layout -t "$SESSION_NAME:Logs" even-vertical

tmux select-pane -t "$SESSION_NAME:Logs.0" -T "Trainer"
tmux select-pane -t "$SESSION_NAME:Logs.1" -T "Orchestrator"
tmux select-pane -t "$SESSION_NAME:Logs.2" -T "Envs"
tmux select-pane -t "$SESSION_NAME:Logs.3" -T "Inference"

tmux send-keys -t "$SESSION_NAME:Logs.0" "$(tail_cmd trainer.log)" C-m
tmux send-keys -t "$SESSION_NAME:Logs.1" "$(tail_cmd orchestrator.log)" C-m
tmux send-keys -t "$SESSION_NAME:Logs.2" "$(tail_cmd 'envs/*/*.log')" C-m
tmux send-keys -t "$SESSION_NAME:Logs.3" "$(tail_cmd inference.log)" C-m

# Window 2: SUCCESS - grep SUCCESS on orch and trainer logs (two stacked panes)
tmux new-window -t "$SESSION_NAME" -n "SUCCESS"

tmux split-window -v -t "$SESSION_NAME:SUCCESS.0"
tmux select-layout -t "$SESSION_NAME:SUCCESS" even-vertical

tmux select-pane -t "$SESSION_NAME:SUCCESS.0" -T "Orchestrator"
tmux select-pane -t "$SESSION_NAME:SUCCESS.1" -T "Trainer"

tmux send-keys -t "$SESSION_NAME:SUCCESS.0" "$(tail_cmd orchestrator.log SUCCESS)" C-m
tmux send-keys -t "$SESSION_NAME:SUCCESS.1" "$(tail_cmd trainer.log SUCCESS)" C-m

# Window 3: Agent (claude code or codex) with log context
tmux new-window -t "$SESSION_NAME" -n "Agent"

AGENT_PROMPT="You are monitoring a prime-rl training run. The output directory is ${OUTPUT_DIR}. Each run writes its artifacts to a run directory (<output_dir>/<run_name>, name auto-generated unless set via --run.name) and its logs to <run_dir>/logs/latest. Find the active run's log directory with: ls -td ${LOG_DIR_GLOB} 2>/dev/null | head -1. Log files relative to it:
  Trainer:        trainer.log
  All nodes:      trainer/node_*.log
  All ranks:      trainer/torchrun/*/*/*/*.log
  Orchestrator:   orchestrator.log
  Inference:      inference.log
  Envs:           envs/*/*.log
  Train envs:     envs/train/*.log
You are running inside tmux session \"${SESSION_NAME}\". The Launcher window (window 0) is where the user runs launch commands. You can read its contents with: tmux capture-pane -t ${SESSION_NAME}:Launcher -p
Help the user monitor and debug this run."

case "$AGENT" in
  claude)
    tmux send-keys -t "$SESSION_NAME:Agent" \
      "claude --permission-mode auto --append-system-prompt \"${AGENT_PROMPT}\"" C-m
    ;;
  codex)
    tmux send-keys -t "$SESSION_NAME:Agent" \
      "codex --yolo \"${AGENT_PROMPT}\"" C-m
    ;;
  *)
    echo "Error: unknown agent '$AGENT' (expected claude|codex)" >&2
    exit 1
    ;;
esac

# Pane title styling
tmux set-option -t "$SESSION_NAME" -g pane-border-status top
tmux set-option -t "$SESSION_NAME" -g pane-border-format " #{pane_title} "

# Focus launcher window and attach
tmux select-window -t "$SESSION_NAME:Launcher"
exec tmux attach-session -t "$SESSION_NAME"
