#!/bin/bash
set -e

# Set higher ulimit for file descriptors to prevent API timeout issues
ulimit -n 32000 2>/dev/null || echo "Warning: Could not set ulimit (may need --ulimit flag in docker run)"

# Allow runtime override of the prime-rl source itself.
# PRIME_RL_REF can be a git tag, branch, or commit hash. PRIME_RL_REPO
# optionally points at a fork; defaults to the upstream repo.
#
# We clone into a per-ref dir under /tmp, seed the venv from the baked
# /app/.venv so heavy wheels (flash-attn, mamba-ssm, …) survive, then
# `uv sync --inexact` re-installs prime_rl from the override source.
# Workdir + PATH swap so the chart's `uv run trainer/inference/orchestrator`
# resolves the entrypoints from the override venv.
if [ -n "$PRIME_RL_REF" ]; then
    PRIME_RL_REPO="${PRIME_RL_REPO:-https://github.com/PrimeIntellect-ai/prime-rl.git}"
    # Refs can contain `/` (e.g. branch names like `feat/foo`); slugify for
    # the cache dir name. Real ref is kept verbatim for git commands.
    REF_SLUG="${PRIME_RL_REF//\//-}"
    DEST="/tmp/prime-rl-${REF_SLUG}"
    if [ ! -d "$DEST/.git" ]; then
        echo "[prime-rl] cloning ${PRIME_RL_REPO} for ${PRIME_RL_REF}"
        rm -rf "$DEST"
        git clone "$PRIME_RL_REPO" "$DEST"
    fi
    # Always fetch + checkout so mutable refs (branches/tags) pick up new
    # commits between pod restarts. No-op for immutable SHAs.
    echo "[prime-rl] refreshing ${PRIME_RL_REF}"
    git -C "$DEST" fetch --quiet --tags --force origin
    git -C "$DEST" checkout --quiet --force "$PRIME_RL_REF"
    # Fast-forward to upstream tip when PRIME_RL_REF is a branch name.
    # Silently no-ops for SHAs/tags (no `origin/<sha>` exists).
    git -C "$DEST" reset --hard --quiet "origin/${PRIME_RL_REF}" 2>/dev/null || true
    if [ ! -d "$DEST/.venv" ]; then
        # Seed from the baked venv so the heavy wheels (flash-attn,
        # mamba-ssm, …) don't have to be rebuilt. Hardlink-copy when /tmp
        # and /app share a filesystem; full copy otherwise.
        cp -al /app/.venv "$DEST/.venv" 2>/dev/null || cp -a /app/.venv "$DEST/.venv"
    fi
    echo "[prime-rl] running uv sync --inexact (this may take a few minutes on cold checkout)"
    ( cd "$DEST" && uv sync --inexact --no-dev )
    export VIRTUAL_ENV="$DEST/.venv"
    export PATH="$DEST/.venv/bin:$PATH"
    cd "$DEST"
fi

# Allow runtime override of the verifiers package version.
# VERIFIERS_VERSION can be a git tag, branch, or commit hash. Runs after the
# PRIME_RL_REF swap so when both are set the install lands in the override
# venv (uv pip install targets $VIRTUAL_ENV when exported).
if [ -n "$VERIFIERS_VERSION" ]; then
    echo "Installing verifiers version: $VERIFIERS_VERSION"
    uv pip install --reinstall-package verifiers \
        "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@${VERIFIERS_VERSION}"
fi

# Execute the main command
exec "$@"
