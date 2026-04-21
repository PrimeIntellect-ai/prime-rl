#!/bin/bash
set -e

# Set higher ulimit for file descriptors to prevent API timeout issues
ulimit -n 32000 2>/dev/null || echo "Warning: Could not set ulimit (may need --ulimit flag in docker run)"

# Allow runtime override of the verifiers package version.
# VERIFIERS_VERSION can be a git tag, branch, or commit hash.
if [ -n "$VERIFIERS_VERSION" ]; then
    echo "Installing verifiers version: $VERIFIERS_VERSION"
    uv pip install --reinstall-package verifiers \
        "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@${VERIFIERS_VERSION}"
fi

# Execute the main command
exec "$@"
