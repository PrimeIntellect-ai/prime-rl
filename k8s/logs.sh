#!/bin/bash

# Default values
RELEASE_NAME=""
SESSION_NAME=""
NAMESPACE="default"

# Help message
show_help() {
  cat << EOF
Usage: $0 [OPTIONS] [RELEASE_NAME]

Opens kubectl logs -f for all three pods (trainer, orchestrator, inference)
in a tmux session with three panes.

OPTIONS:
  -h, --help            Show this help message
  -s, --session-name    Custom tmux session name (default: k8s-<release>)
  -n, --namespace       Kubernetes namespace (default: default)

ARGUMENTS:
  RELEASE_NAME          Helm release name (required)

EXAMPLES:
  $0 reverse-text
  $0 -n my-namespace my-release
  $0 --session-name my-logs reverse-text
EOF
}

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -s|--session-name)
      if [[ -z "$2" ]]; then
        echo "Error: --session-name requires a value" >&2
        exit 1
      fi
      SESSION_NAME="$2"
      shift 2
      ;;
    -n|--namespace)
      if [[ -z "$2" ]]; then
        echo "Error: --namespace requires a value" >&2
        exit 1
      fi
      NAMESPACE="$2"
      shift 2
      ;;
    -*)
      echo "Error: Unknown option: $1" >&2
      show_help
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

# Get release name from positional argument
if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  RELEASE_NAME="${POSITIONAL[0]}"
fi

# Validate required arguments
if [[ -z "$RELEASE_NAME" ]]; then
  echo "Error: RELEASE_NAME is required" >&2
  show_help
  exit 1
fi

# Set default session name if not provided
if [[ -z "$SESSION_NAME" ]]; then
  SESSION_NAME="k8s-${RELEASE_NAME}"
fi

# Find pods
echo "Looking for pods with release name: ${RELEASE_NAME} in namespace: ${NAMESPACE}"
TRAINER_POD=$(kubectl get pods -n "$NAMESPACE" -l "role=trainer,app.kubernetes.io/instance=${RELEASE_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
ORCHESTRATOR_POD=$(kubectl get pods -n "$NAMESPACE" -l "role=orchestrator,app.kubernetes.io/instance=${RELEASE_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
INFERENCE_POD=$(kubectl get pods -n "$NAMESPACE" -l "role=inference,app.kubernetes.io/instance=${RELEASE_NAME}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

# Check if pods were found
if [[ -z "$TRAINER_POD" ]] || [[ -z "$ORCHESTRATOR_POD" ]] || [[ -z "$INFERENCE_POD" ]]; then
  echo "Error: Could not find all required pods" >&2
  echo "Trainer: ${TRAINER_POD:-NOT FOUND}" >&2
  echo "Orchestrator: ${ORCHESTRATOR_POD:-NOT FOUND}" >&2
  echo "Inference: ${INFERENCE_POD:-NOT FOUND}" >&2
  exit 1
fi

echo "Found pods:"
echo "  Trainer: ${TRAINER_POD}"
echo "  Orchestrator: ${ORCHESTRATOR_POD}"
echo "  Inference: ${INFERENCE_POD}"
echo ""

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Attaching to existing tmux session: $SESSION_NAME"
  exec tmux attach-session -t "$SESSION_NAME"
else
  echo "Creating new tmux session: $SESSION_NAME"

  # Start new tmux session with first window
  tmux new-session -d -s "$SESSION_NAME" -n "Logs"

  # Split into 3 vertical panes
  tmux split-window -v -t "$SESSION_NAME:Logs.0"
  tmux split-window -v -t "$SESSION_NAME:Logs.1"
  tmux select-layout -t "$SESSION_NAME:Logs" even-vertical

  # Set pane titles
  tmux select-pane -t "$SESSION_NAME:Logs.0" -T "Trainer"
  tmux select-pane -t "$SESSION_NAME:Logs.1" -T "Orchestrator"
  tmux select-pane -t "$SESSION_NAME:Logs.2" -T "Inference"

  # Start kubectl logs -f in each pane
  # Pass KUBECONFIG if it's set
  KUBECONFIG_CMD=""
  if [[ -n "$KUBECONFIG" ]]; then
    KUBECONFIG_CMD="export KUBECONFIG='$KUBECONFIG'; "
  fi

  tmux send-keys -t "$SESSION_NAME:Logs.0" "${KUBECONFIG_CMD}kubectl logs -f -n $NAMESPACE $TRAINER_POD" C-m
  tmux send-keys -t "$SESSION_NAME:Logs.1" "${KUBECONFIG_CMD}kubectl logs -f -n $NAMESPACE $ORCHESTRATOR_POD" C-m
  tmux send-keys -t "$SESSION_NAME:Logs.2" "${KUBECONFIG_CMD}kubectl logs -f -n $NAMESPACE $INFERENCE_POD" C-m

  # Enable pane titles
  tmux set-option -t "$SESSION_NAME" -g pane-border-status top
  tmux set-option -t "$SESSION_NAME" -g pane-border-format " #{pane_title} "

  # Focus trainer pane and attach
  tmux select-pane -t "$SESSION_NAME:Logs.0"
  exec tmux attach-session -t "$SESSION_NAME"
fi
