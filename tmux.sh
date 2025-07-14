#!/bin/bash

# Default session name
DEFAULT_EXPERIMENT_ID="rl"

# Parse arguments
EXPERIMENT_NAME="$DEFAULT_EXPERIMENT_ID"
KILL_SESSION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        kill)
            KILL_SESSION=true
            shift
            ;;
        *)
            # If it's not 'kill', treat it as session name
            EXPERIMENT_NAME="$1"
            shift
            ;;
    esac
done

# Handle kill command
if [ "$KILL_SESSION" = true ]; then
    if tmux has-session -t "$EXPERIMENT_NAME" 2>/dev/null; then
        echo "Killing tmux session: $EXPERIMENT_NAME"
        tmux kill-session -t "$EXPERIMENT_NAME"
        echo "Session '$EXPERIMENT_NAME' terminated."
    else
        echo "Session '$EXPERIMENT_NAME' not found."
    fi
    exit 0
fi

# Check if we're already inside a tmux session
if [ -n "$TMUX" ]; then
    # We're inside tmux, so switch to the session instead of attaching
    if tmux has-session -t "$EXPERIMENT_NAME" 2>/dev/null; then
        echo "Session '$EXPERIMENT_NAME' already exists. Switching..."
        tmux switch-client -t "$EXPERIMENT_NAME"
    else
        echo "Creating new tmux session: $EXPERIMENT_NAME"
        tmux new-session -d -s "$EXPERIMENT_NAME" -n "RL"
        
        # Window 1: RL - Create the layout first
        # Create 2 more panes for a total of 3
        tmux split-window -v -t "$EXPERIMENT_NAME:RL.0"
        tmux split-window -v -t "$EXPERIMENT_NAME:RL.1"
        
        # Apply even-vertical layout to distribute panes evenly
        tmux select-layout -t "$EXPERIMENT_NAME:RL" even-vertical
        
        # Set pane titles
        tmux select-pane -t "$EXPERIMENT_NAME:RL.0" -T "Trainer"
        tmux select-pane -t "$EXPERIMENT_NAME:RL.1" -T "Orchestrator"
        tmux select-pane -t "$EXPERIMENT_NAME:RL.2" -T "Inference"
        
        # Send commands to panes
        # Pane 1: Trainer - stays empty
        
        # Pane 2: Orchestrator
        tmux send-keys -t "$EXPERIMENT_NAME:RL.1" 'while true; do
  echo "Waiting for orchestrator log file..."
  while [ ! -f logs/'"$EXPERIMENT_NAME"'/orchestrator.log ]; do sleep 1; done
  echo "Following orchestrator.log..."
  tail -F logs/'"$EXPERIMENT_NAME"'/orchestrator.log
done' C-m
        
        # Pane 3: Inference
        tmux send-keys -t "$EXPERIMENT_NAME:RL.2" 'while true; do
  echo "Waiting for inference log file..."
  while [ ! -f logs/'"$EXPERIMENT_NAME"'/inference.log ]; do sleep 1; done
  echo "Following inference.log..."
  tail -F logs/'"$EXPERIMENT_NAME"'/inference.log
done' C-m
        
        # Create second window
        tmux new-window -t "$EXPERIMENT_NAME:2" -n "Monitor"
        
        # Window 2: Monitor - Create horizontal split
        tmux split-window -h -t "$EXPERIMENT_NAME:Monitor"
        tmux select-layout -t "$EXPERIMENT_NAME:Monitor" even-horizontal
        
        # Set pane titles and run commands
        tmux select-pane -t "$EXPERIMENT_NAME:Monitor.0" -T "GPU"
        tmux send-keys -t "$EXPERIMENT_NAME:Monitor.0" "nvtop" C-m
        
        tmux select-pane -t "$EXPERIMENT_NAME:Monitor.1" -T "CPU"
        tmux send-keys -t "$EXPERIMENT_NAME:Monitor.1" "htop" C-m
        
        # Enable pane titles for all windows
        tmux set-option -t "$EXPERIMENT_NAME" -g pane-border-status top
        tmux set-option -t "$EXPERIMENT_NAME" -g pane-border-format " #{pane_title} "
        
        # Also set for each window explicitly
        tmux set-window-option -t "$EXPERIMENT_NAME:RL" pane-border-status top
        tmux set-window-option -t "$EXPERIMENT_NAME:Monitor" pane-border-status
        
        # Select first window and first pane
        tmux select-window -t "$EXPERIMENT_NAME:RL"
        tmux select-pane -t "$EXPERIMENT_NAME:RL.0"
        
        # Switch to the new session
        tmux switch-client -t "$EXPERIMENT_NAME"
    fi
else
    # Not inside tmux, use attach
    if tmux has-session -t "$EXPERIMENT_NAME" 2>/dev/null; then
        echo "Session '$EXPERIMENT_NAME' already exists. Attaching..."
        tmux attach-session -t "$EXPERIMENT_NAME"
    else
        echo "Creating new tmux session: $EXPERIMENT_NAME"
        
        # Start new tmux session with first window
        tmux new-session -d -s "$EXPERIMENT_NAME" -n "RL"
        
        # Window 1: RL - Create 2 more panes for a total of 3
        tmux split-window -v -t "$EXPERIMENT_NAME:RL.0"
        tmux split-window -v -t "$EXPERIMENT_NAME:RL.1"
        
        # Apply even-vertical layout
        tmux select-layout -t "$EXPERIMENT_NAME:RL" even-vertical
        
        # Set pane titles
        tmux select-pane -t "$EXPERIMENT_NAME:RL.0" -T "Trainer"
        tmux select-pane -t "$EXPERIMENT_NAME:RL.1" -T "Orchestrator"
        tmux select-pane -t "$EXPERIMENT_NAME:RL.2" -T "Inference"
        
        # Send commands to panes
        # Pane 2: Orchestrator
        tmux send-keys -t "$EXPERIMENT_NAME:RL.1" 'while true; do
  echo "Waiting for orchestrator log file..."
  while [ ! -f logs/'"$EXPERIMENT_NAME"'/orchestrator.log ]; do sleep 1; done
  echo "Following orchestrator.log..."
  tail -F logs/'"$EXPERIMENT_NAME"'/orchestrator.log
done' C-m
        
        # Pane 3: Inference
        tmux send-keys -t "$EXPERIMENT_NAME:RL.2" 'while true; do
  echo "Waiting for inference log file..."
  while [ ! -f logs/'"$EXPERIMENT_NAME"'/inference.log ]; do sleep 1; done
  echo "Following inference.log..."
  tail -F logs/'"$EXPERIMENT_NAME"'/inference.log
done' C-m
        
        # Create second window
        tmux new-window -t "$EXPERIMENT_NAME" -n "Monitor"
        
        # Window 2: Monitor
        tmux split-window -h -t "$EXPERIMENT_NAME:Monitor"
        tmux select-layout -t "$EXPERIMENT_NAME:Monitor" even-horizontal
        
        # Set pane titles and run commands
        tmux select-pane -t "$EXPERIMENT_NAME:Monitor.0" -T "GPU"
        tmux send-keys -t "$EXPERIMENT_NAME:Monitor.0" "nvtop" C-m
        
        tmux select-pane -t "$EXPERIMENT_NAME:Monitor.1" -T "CPU"
        tmux send-keys -t "$EXPERIMENT_NAME:Monitor.1" "htop" C-m
        
        # Enable pane titles for all windows
        tmux set-option -t "$EXPERIMENT_NAME" -g pane-border-status top
        tmux set-option -t "$EXPERIMENT_NAME" -g pane-border-format " #{pane_title} "
        
        # Also set for each window explicitly
        tmux set-window-option -t "$EXPERIMENT_NAME:RL" pane-border-status top
        tmux set-window-option -t "$EXPERIMENT_NAME:Monitor" pane-border-status top
        
        # Select first window and first pane
        tmux select-window -t "$EXPERIMENT_NAME:RL"
        tmux select-pane -t "$EXPERIMENT_NAME:RL.0"
        
        # Attach to the session
        tmux attach-session -t "$EXPERIMENT_NAME"
    fi
fi 