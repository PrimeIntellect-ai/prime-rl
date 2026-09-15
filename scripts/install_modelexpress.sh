#!/usr/bin/env bash
# Build the ModelExpress metadata server and Redis backend used by SLURM RL jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODELEXPRESS_REPOSITORY="https://github.com/ai-dynamo/modelexpress.git"
MODELEXPRESS_REF="v0.3.0"
REDIS_VERSION="7.4.2"
REDIS_SHA256="4ddebbf09061cbb589011786febdb34f29767dd7f89dbe712d2b68e808af6a1f"

if [[ $# -gt 1 ]]; then
    echo "Usage: $(basename "$0") [modelexpress-ref]" >&2
    exit 1
elif [[ $# -eq 1 ]]; then
    # mx_refit needs a server carrying RefitService, which the default tag does
    # not have. Callers must be able to build the server from the same commit as
    # the client, or the pair fails with UNIMPLEMENTED.
    MODELEXPRESS_REF="$1"
fi

BIN_DIR="$PROJECT_DIR/third_party/modelexpress/bin"
mkdir -p "$BIN_DIR"

# Record the built commit rather than inferring identity from --version, which
# reports a package version and cannot distinguish two commits on one tag.
SOURCE_STAMP="$BIN_DIR/modelexpress-server.source-sha"

if [[ -x "$BIN_DIR/modelexpress-server" && -f "$SOURCE_STAMP" ]] \
    && [[ "$(cat "$SOURCE_STAMP")" == "$MODELEXPRESS_REF" || "$(cut -c1-40 "$SOURCE_STAMP")" == "$MODELEXPRESS_REF" ]]; then
    echo "modelexpress-server $MODELEXPRESS_REF already installed at $BIN_DIR"
else
    command -v cargo >/dev/null || {
        echo "cargo not found; install Rust 1.90 or newer before running this script" >&2
        exit 1
    }
    command -v protoc >/dev/null || {
        echo "protoc not found; install Protocol Buffers before running this script" >&2
        exit 1
    }
    BUILD_DIR=$(mktemp -d)
    trap 'rm -rf "$BUILD_DIR"' EXIT
    # `clone --branch` takes a branch or tag only, so fetch the ref explicitly:
    # mx_refit needs a server built from a specific commit, and a tag cannot
    # name one unambiguously. MODELEXPRESS_REPOSITORY may be a local path or
    # bundle, so a preserved source tree can be built without pushing it.
    git init -q "$BUILD_DIR/modelexpress"
    (
        cd "$BUILD_DIR/modelexpress"
        git remote add origin "$MODELEXPRESS_REPOSITORY"
        git fetch --depth 1 origin "$MODELEXPRESS_REF" \
            || git fetch origin "$MODELEXPRESS_REF" \
            || {
                echo "Could not fetch ModelExpress ref '$MODELEXPRESS_REF' from $MODELEXPRESS_REPOSITORY" >&2
                exit 1
            }
        git checkout -q FETCH_HEAD
        RESOLVED=$(git rev-parse HEAD)
        case "$MODELEXPRESS_REF" in
            "$RESOLVED"|"${RESOLVED:0:7}"*)
                ;;
            *)
                echo "Built ModelExpress $MODELEXPRESS_REF resolves to $RESOLVED" >&2
                ;;
        esac
        echo "$RESOLVED" > "$BUILD_DIR/source-sha"
        cargo build --release --bin modelexpress-server
    )
    cp "$BUILD_DIR/modelexpress/target/release/modelexpress-server" "$BIN_DIR/"
    cp "$BUILD_DIR/source-sha" "$SOURCE_STAMP"
    echo "modelexpress-server built from $(cat "$SOURCE_STAMP")"
fi

if [[ -x "$BIN_DIR/redis-server" ]] \
    && "$BIN_DIR/redis-server" --version | grep -q "v=$REDIS_VERSION"; then
    echo "redis-server $REDIS_VERSION already installed at $BIN_DIR"
else
    BUILD_DIR="${BUILD_DIR:-$(mktemp -d)}"
    trap 'rm -rf "$BUILD_DIR"' EXIT
    REDIS_ARCHIVE="$BUILD_DIR/redis-${REDIS_VERSION}.tar.gz"
    curl --fail --location --silent --show-error \
        --output "$REDIS_ARCHIVE" \
        "https://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz"
    echo "$REDIS_SHA256  $REDIS_ARCHIVE" | sha256sum --check
    tar -xzf "$REDIS_ARCHIVE" -C "$BUILD_DIR"
    make -C "$BUILD_DIR/redis-${REDIS_VERSION}" -j redis-server MALLOC=libc
    cp "$BUILD_DIR/redis-${REDIS_VERSION}/src/redis-server" "$BIN_DIR/"
fi

echo "Installed ModelExpress server dependencies in $BIN_DIR"
