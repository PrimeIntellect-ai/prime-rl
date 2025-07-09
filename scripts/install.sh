#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

main() {
    # Check if sudo is installed
    if ! command -v sudo &> /dev/null; then
        apt update
        apt install sudo -y
    fi

    log_info "Updating apt..."
    sudo apt update

    log_info "Installing git, tmux, htop, nvtop, cmake, python3-dev, cgroup-tools..."
    sudo apt install git tmux htop nvtop cmake python3-dev cgroup-tools -y

    log_info "Cloning repository..."
    git clone git@github.com:PrimeIntellect-ai/prime-rl.git

    log_info "Entering project directory..."
    cd prime-rl

    log_info "Installing gsutil..."
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
    sudo apt-get update -y
    sudo apt-get install -y google-cloud-cli

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi

    log_info "Installing dependencies in virtual environment..."
    uv sync && uv sync --extra fa
    log_info "Installation completed!"
}

main
