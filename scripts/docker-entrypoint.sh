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

# Allow runtime override of the prime-rl source itself.
# PRIME_RL_REF can be a git tag, branch, or commit hash. PRIME_RL_REPO
# optionally points at a fork; defaults to the upstream repo.
#
# We clone into a per-ref dir under /tmp, hardlink-copy the baked /app/.venv
# so the heavy wheels (flash-attn, mamba-ssm, …) stay put, then `uv sync
# --inexact` re-installs prime_rl from the override source. Workdir + PATH
# swap so the chart's `uv run trainer/inference/orchestrator` resolves the
# entrypoints from the override venv.
if [ -n "$PRIME_RL_REF" ]; then
    PRIME_RL_REPO="${PRIME_RL_REPO:-https://github.com/PrimeIntellect-ai/prime-rl.git}"
    DEST="/tmp/prime-rl-${PRIME_RL_REF}"
    if [ ! -d "$DEST/.git" ]; then
        echo "[prime-rl] checking out ${PRIME_RL_REF} from ${PRIME_RL_REPO}"
        rm -rf "$DEST"
        git clone "$PRIME_RL_REPO" "$DEST"
        git -C "$DEST" checkout "$PRIME_RL_REF"
        # Seed the override venv from the baked one to keep flash-attn etc.;
        # fall back to a plain copy when /tmp and /app are on different
        # filesystems (hardlinks across devices fail).
        cp -al /app/.venv "$DEST/.venv" 2>/dev/null || cp -a /app/.venv "$DEST/.venv"
        echo "[prime-rl] running uv sync --inexact (this may take a few minutes)"
        ( cd "$DEST" && uv sync --inexact --no-dev )
    else
        echo "[prime-rl] reusing cached checkout at ${DEST}"
    fi
    export VIRTUAL_ENV="$DEST/.venv"
    export PATH="$DEST/.venv/bin:$PATH"
    cd "$DEST"
fi

# Execute the main command
exec "$@"
