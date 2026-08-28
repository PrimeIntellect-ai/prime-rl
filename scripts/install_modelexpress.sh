#!/usr/bin/env bash
# Build the ModelExpress metadata server and Redis backend used by SLURM RL jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODELEXPRESS_REPOSITORY="https://github.com/ai-dynamo/modelexpress.git"
MODELEXPRESS_REF="v0.3.0"
REDIS_VERSION="7.4.2"
REDIS_SHA256="4ddebbf09061cbb589011786febdb34f29767dd7f89dbe712d2b68e808af6a1f"

if [[ $# -gt 0 ]]; then
    echo "This installer does not accept arguments" >&2
    exit 1
fi

BIN_DIR="$PROJECT_DIR/third_party/modelexpress/bin"
mkdir -p "$BIN_DIR"

if [[ -x "$BIN_DIR/modelexpress-server" ]] \
    && "$BIN_DIR/modelexpress-server" --version 2>/dev/null | grep -q "${MODELEXPRESS_REF#v}"; then
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
    git clone --depth 1 --branch "$MODELEXPRESS_REF" "$MODELEXPRESS_REPOSITORY" "$BUILD_DIR/modelexpress"
    (
        cd "$BUILD_DIR/modelexpress"
        cargo build --release --bin modelexpress-server
    )
    cp "$BUILD_DIR/modelexpress/target/release/modelexpress-server" "$BIN_DIR/"
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
