#!/bin/bash
set -e

# Set higher ulimit for file descriptors to prevent API timeout issues
ulimit -n 32000 2>/dev/null || echo "Warning: Could not set ulimit (may need --ulimit flag in docker run)"

# Allow runtime override of the prime-rl source itself.
# PRIME_RL_REF can be a git tag, branch, or full commit hash. PRIME_RL_REPO
# optionally points at a fork; defaults to the upstream repo.
#
# The ref is fetched shallowly under /tmp and synced INTO the baked
# /app/.venv: uv re-points the first-party editables (prime_rl, verifiers,
# renderers, ...) at the checkout and reinstalls only third-party pins
# that differ from the image's lockfile, so the heavy prebuilt wheels
# (flash-attn, deep-ep, mamba-ssm, ...) are never rebuilt. A container
# starts from a fresh filesystem, so mutating /app/.venv in place is safe.
# The pods of the prime-rl-fft chart run this with a sha pinned by the
# dispatcher; the exported PRIME_RL_SOURCE_COMMIT records what ran.
if [ -n "$PRIME_RL_REF" ]; then
    PRIME_RL_REPO="${PRIME_RL_REPO:-https://github.com/PrimeIntellect-ai/prime-rl.git}"
    # Slug + content hash for the checkout dir name. Slug keeps the path
    # human-readable; the hash (over repo + ref) prevents collisions
    # between distinct refs that slugify the same way (e.g. `feat/foo`
    # vs `feat-foo`) and between the same ref on different forks.
    REF_SLUG="${PRIME_RL_REF//\//-}"
    REF_HASH=$(echo -n "${PRIME_RL_REPO}|${PRIME_RL_REF}" | md5sum | cut -c1-12)
    DEST="/tmp/prime-rl-${REF_SLUG}-${REF_HASH}"
    # Rewrite git@github.com URLs to https so submodules listed with SSH
    # URLs (deps/verifiers, deps/renderers, deps/prime-envs) clone from the
    # pod without ssh keys. Via GIT_CONFIG_* rather than `git config
    # --global`: the pod's HOME may not be writable by its uid.
    export GIT_CONFIG_COUNT=1
    export GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf"
    export GIT_CONFIG_VALUE_0="git@github.com:"
    rm -rf "$DEST"
    git init --quiet "$DEST"
    git -C "$DEST" remote add origin "$PRIME_RL_REPO"
    # Depth-1 fetch of the ref itself: GitHub serves branches, tags and
    # full commit shas alike (abbreviated shas are not fetchable), and
    # prime-rl's history (~700 MB of .git) is not needed to run it.
    echo "[prime-rl] fetching ${PRIME_RL_REF} from ${PRIME_RL_REPO}"
    git -C "$DEST" fetch --quiet --depth 1 origin "$PRIME_RL_REF"
    git -C "$DEST" checkout --quiet --force --detach FETCH_HEAD
    # Same public submodules the image build inits (build_image.yaml);
    # a pinned commit that isn't the submodule's branch tip is fetched by
    # sha, which GitHub allows.
    git -C "$DEST" submodule sync --quiet --recursive
    git -C "$DEST" submodule update --quiet --init --recursive --depth 1 -- \
        deps/pydantic-config deps/verifiers deps/renderers deps/prime-envs
    PRIME_RL_SOURCE_COMMIT=$(git -C "$DEST" rev-parse HEAD)
    export PRIME_RL_SOURCE_COMMIT
    echo "[prime-rl] source overlay at ${PRIME_RL_SOURCE_COMMIT}"

    # Give the verifiers/renderers editables the same versions the image
    # build does (Dockerfile.cuda): resolve them with the repo's script
    # (git describe, else the highest `>=` floor any pyproject declares),
    # then bake them into hatch-vcs's fallback-version and drop the
    # submodules' git metadata so the fallback is what gets used. The
    # shallow clones carry no tags, so hatch-vcs would otherwise stamp
    # them 0.0.1.dev1 and every `verifiers>=…` floor in the tree would be
    # unmet, letting the CLI install below swap the editable for a wheel.
    if [ -f "$DEST/scripts/docker-editable-pretend-versions.sh" ]; then
        eval "$(bash "$DEST/scripts/docker-editable-pretend-versions.sh" --shell "$DEST")"
        rm -f "$DEST/deps/verifiers/.git" "$DEST/deps/renderers/.git"
        sed -i "s/fallback-version = \"0.0.0\"/fallback-version = \"${VERIFIERS_PRETEND_VERSION}\"/" "$DEST/deps/verifiers/pyproject.toml"
        sed -i "s/fallback-version = \"0.0.0\"/fallback-version = \"${RENDERERS_PRETEND_VERSION}\"/" "$DEST/deps/renderers/pyproject.toml"
    fi

    # Wheels the in-place sync cannot rebuild: a lockfile that pins any of
    # these differently from the image needs a real image build. Fail here
    # rather than minutes later inside a half-installed torch. Stdlib only;
    # --project /app pins the interpreter to the baked venv whatever the cwd.
    uv run --no-sync --project /app python - "$DEST/uv.lock" /app/uv.lock <<'HEAVY_PINS_PY'
import sys
import tomllib

HEAVY = {
    "torch", "torchvision", "torchaudio", "triton", "vllm", "vllm-router",
    "flash-attn", "flash-attn-3", "flashinfer-python", "tilelang",
    "deep-ep", "deep-gemm", "mamba-ssm", "nixl", "prime-kernels", "quack-kernels",
}


def pins(path):
    with open(path, "rb") as f:
        lock = tomllib.load(f)
    out = {}
    for pkg in lock.get("package", []):
        if pkg.get("name") in HEAVY:
            source = repr(sorted((pkg.get("source") or {}).items()))
            out.setdefault(pkg["name"], set()).add((pkg.get("version"), source))
    return out


ref_pins, image_pins = pins(sys.argv[1]), pins(sys.argv[2])
drift = sorted(name for name in HEAVY if ref_pins.get(name) != image_pins.get(name))
if drift:
    print(
        f"[prime-rl] {sys.argv[1]} pins {', '.join(drift)} differently from the image; "
        "the runtime source overlay cannot rebuild these. Build an image for this ref "
        "(build_image.yaml) or run it on an image whose lockfile matches.",
        file=sys.stderr,
    )
    sys.exit(1)
HEAVY_PINS_PY

    # Sync the checkout into the baked venv. Same extras as Dockerfile.cuda
    # so an unchanged lockfile is a no-op for third-party packages (keep
    # the two lists in sync). --inexact leaves packages the checkout
    # doesn't declare in place instead of uninstalling them.
    export UV_PROJECT_ENVIRONMENT=/app/.venv
    export VIRTUAL_ENV=/app/.venv
    echo "[prime-rl] syncing ${DEST} into /app/.venv"
    ( cd "$DEST" && uv sync --inexact --locked --no-dev --all-packages \
        --extra gpu --extra dashboard --extra flash-attn --extra flash-attn-3 \
        --extra flash-attn-cute --extra disagg --extra quack --extra kernels \
        --group mamba-ssm )
    # Replay the image's post-sync step (Dockerfile.cuda; keep the two in
    # sync): the prime CLI pulls prime-traces / prime-sandboxes / openai
    # past the lock's pins, and the --locked sync above just rolled them
    # back, which leaves `prime env install` unable to import.
    #
    # FIXME(prime#940): workaround, remove once the prime CLI drops
    # verifiers as a dependency
    # (https://github.com/PrimeIntellect-ai/prime/pull/940). The CLI pins
    # verifiers==0.2.0, which would swap the editable for the PyPI wheel
    # and break prime-rl's imports; the --overrides file ignores that one
    # requirement. When removing it, drop the overrides file and the
    # --overrides flag here and in Dockerfile.cuda.
    echo "[prime-rl] installing the prime CLI"
    printf 'verifiers\n' > /tmp/prime-cli-overrides.txt
    uv pip install --overrides /tmp/prime-cli-overrides.txt 'prime>=0.7'
    # The chart's `uv run --no-sync <entrypoint>` commands resolve the
    # project from the cwd; the venv itself stays /app/.venv via
    # UV_PROJECT_ENVIRONMENT above.
    cd "$DEST"
fi

# Allow runtime override of the verifiers package version.
# VERIFIERS_VERSION can be a git tag, branch, or commit hash. Runs after the
# PRIME_RL_REF overlay so when both are set verifiers is installed last
# (uv pip install targets $VIRTUAL_ENV, which is /app/.venv either way).
if [ -n "$VERIFIERS_VERSION" ]; then
    echo "Installing verifiers version: $VERIFIERS_VERSION"
    uv pip install --reinstall-package verifiers \
        "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@${VERIFIERS_VERSION}"
fi

# Execute the main command
exec "$@"
