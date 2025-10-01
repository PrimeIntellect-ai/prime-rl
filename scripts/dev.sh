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
if [ ! -d ~/verifiers ]; then
    log_info "Setting up verifiers locally..."
    cd ~ && git clone https://github.com/PrimeIntellect-ai/verifiers.git && cd verifiers && uv sync --extra dev && cd -
fi

if [ ! -d ~/prime-environments ]; then
    log_info "Setting up prime-environments locally..."
    cd ~ && curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-environments/main/scripts/install.sh | bash && cd -
fi

log_info "Installing verifiers locally..."
uv pip install -e ~/verifiers

log_info "Turned on dev mode! To turn it off, run \`bash scripts/dev.sh off\`"