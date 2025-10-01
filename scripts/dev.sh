#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Turn off dev mode
if [ "$1" == "off" ]; then
    log_info "Turning off dev mode..."
    uv sync && uv sync --all-extras
    log_info "Turned off dev mode!"
    exit 0
fi

# Turn on dev mode
log_info "Adding verifiers as submodule..."
if [ ! -d "dev/verifiers" ]; then
    git clone git@github.com:primeintellect-ai/verifiers.git dev/verifiers
    uv pip install -e dev/verifiers
fi

log_info "Adding prime-environments as submodule..."
if [ ! -d "dev/prime-environments" ]; then
    git clone git@github.com:primeintellect-ai/prime-environments.git dev/prime-environments
fi

log_info "Turned on dev mode! To turn it off, run \`bash scripts/dev.sh off\`"