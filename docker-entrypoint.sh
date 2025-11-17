#!/bin/bash
set -e

# Set higher ulimit for file descriptors to prevent API timeout issues
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit (may need --ulimit flag in docker run)"

# Execute the main command
exec "$@"
