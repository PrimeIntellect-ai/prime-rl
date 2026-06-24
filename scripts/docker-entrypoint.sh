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
    # Slug + content hash for the cache dir name. Slug keeps the path
    # human-readable; the hash (over repo + ref) prevents collisions
    # between distinct refs that slugify the same way (e.g. `feat/foo`
    # vs `feat-foo`) and between the same ref on different forks.
    REF_SLUG="${PRIME_RL_REF//\//-}"
    REF_HASH=$(echo -n "${PRIME_RL_REPO}|${PRIME_RL_REF}" | md5sum | cut -c1-12)
    DEST="/tmp/prime-rl-${REF_SLUG}-${REF_HASH}"
    # Rewrite ssh://git@github.com URLs to https so submodules listed
    # with SSH URLs (deps/verifiers, deps/renderers, deps/research-envs)
    # can be cloned from the pod without ssh keys.
    git config --global url."https://github.com/".insteadOf "git@github.com:"
    if [ ! -d "$DEST/.git" ]; then
        echo "[prime-rl] cloning ${PRIME_RL_REPO} for ${PRIME_RL_REF}"
        rm -rf "$DEST"
        git clone --recurse-submodules "$PRIME_RL_REPO" "$DEST"
    fi
    # Always fetch + checkout so mutable refs (branches/tags) pick up new
    # commits between pod restarts. No-op for immutable SHAs.
    echo "[prime-rl] refreshing ${PRIME_RL_REF}"
    git -C "$DEST" fetch --quiet --tags --force origin
    git -C "$DEST" checkout --quiet --force "$PRIME_RL_REF"
    # Fast-forward to upstream tip when PRIME_RL_REF is a branch name.
    # Silently no-ops for SHAs/tags (no `origin/<sha>` exists).
    git -C "$DEST" reset --hard --quiet "origin/${PRIME_RL_REF}" 2>/dev/null || true
    # Refresh submodules to whatever the parent commit pins. `sync`
    # picks up URL changes in .gitmodules between checkouts on cache
    # hit; `update --init --recursive` applies the pinned SHAs.
    git -C "$DEST" submodule sync --recursive
    git -C "$DEST" submodule update --init --recursive
    if [ ! -d "$DEST/.venv" ]; then
        # Seed from the baked venv so the heavy wheels (flash-attn,
        # mamba-ssm, …) don't have to be rebuilt. Plain `cp -a` (no
        # hardlinks): a subsequent `uv sync` writes into this tree, and
        # hardlinks would leak those writes back into /app/.venv when
        # /tmp and /app share a filesystem.
        cp -a /app/.venv "$DEST/.venv"
    fi
    echo "[prime-rl] running uv sync --inexact (this may take a few minutes on cold checkout)"
    # Mirror the image's uv sync extras — explicit instead
    # of --all-extras so we don't pull in `disagg` / `quack` and trigger
    # heavy source builds (deep-ep, deep-gemm, quack-kernels) at pod
    # startup. --inexact keeps the seeded venv's pre-built wheels
    # (flash-attn-3, mamba-ssm) in place; uv only rebuilds them if the
    # override's lockfile pins different versions.
    ( cd "$DEST" && uv sync --inexact --no-dev \
        --extra flash-attn --extra flash-attn-3 --extra flash-attn-cute \
        --extra envs --extra gpt-oss --extra modelexpress \
        --group mamba-ssm )
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
