#!/bin/bash
set -e

# Set higher ulimit for file descriptors for sandbox environments (need >= 65536)
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit (may need --ulimit flag in docker run)"

# Execute the main command
exec "$@"
