#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

if [ "$1" == "off" ]; then
    # Turn off dev mode
    uv sync && uv sync --all-extras && unset UV_NO_SYNC
    log_info "Turned off dev mode!"
else
    # Turn on dev mode
    if [ ! -d ~/verifiers ]; then
        log_info "Did not find ~/verifiers. Setting up..."
        cd ~ && curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/scripts/install.sh | bash && cd -
    fi

    if [ ! -d ~/prime-environments ]; then
        log_info "Did not find ~/prime-environments. Setting up..."
        cd ~ && curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-environments/main/scripts/install.sh | bash && cd -
    fi

    # Install verifiers as local, editable package
    uv pip install -e ~/verifiers

    # This is needed so that `uv run ...` does not sync with lock file and remove the temporary packages
    export UV_NO_SYNC=1

    log_info "Turned on dev mode! To add local environments, run 'uv pip install -e path/to/env'. To turn it off, run 'source scripts/dev.sh off'"
fi

set +e
