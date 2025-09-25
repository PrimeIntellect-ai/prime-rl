#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Confirm destructive action
confirm_cleanup() {
    echo "This will remove the following paths recursively:"
    echo "  - **/logs"
    echo "  - **/checkpoints"
    echo "  - **/weights"
    echo "  - **/rollouts"
    echo "  - **/wandb"
    echo "  - **/evals"
    echo "  - **/evals"
    echo "  - .pydantic_config"
    while true; do
        read -r -p "Proceed? [y/N]: " response
        case "$response" in
            [yY]|[yY][eE][sS]) break ;;
            [nN]|[nN][oO]|"") echo "Aborted."; exit 1 ;;
            *) echo "Please answer y or n." ;;
        esac
    done
}

# Remove logs, checkpoints, weights, rollouts, wandb
confirm_cleanup
rm -rf **/logs **/checkpoints **/weights **/rollouts **/wandb **/evals **/torchrun *.pydantic_config
log_info "Cleaned up!"