#!/usr/bin/env bash
# Build the ModelExpress metadata server and Redis backend used by SLURM RL jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODELEXPRESS_REPOSITORY="https://github.com/ai-dynamo/modelexpress.git"
# Must match the modelexpress client rev pinned in pyproject.toml.
MODELEXPRESS_REF="8512b8c7130db34721a0b2ec57c23198fed3ef4f"
PROTOC_VERSION="29.3"
REDIS_VERSION="7.4.2"
REDIS_SHA256="4ddebbf09061cbb589011786febdb34f29767dd7f89dbe712d2b68e808af6a1f"

if [[ $# -gt 0 ]]; then
    echo "This installer does not accept arguments" >&2
    exit 1
fi

MX_DIR="$PROJECT_DIR/third_party/modelexpress"
BIN_DIR="$MX_DIR/bin"
mkdir -p "$BIN_DIR"

if [[ -x "$BIN_DIR/modelexpress-server" ]] && [[ "$(cat "$BIN_DIR/modelexpress-server.ref" 2>/dev/null)" == "$MODELEXPRESS_REF" ]]; then
    echo "modelexpress-server $MODELEXPRESS_REF already installed at $BIN_DIR"
else
    BUILD_DIR=$(mktemp -d)
    trap 'rm -rf "$BUILD_DIR"' EXIT
    # Bootstrap a job-local Rust toolchain and protoc when the host has none.
    if ! command -v cargo >/dev/null; then
        export RUSTUP_HOME="$MX_DIR/rustup" CARGO_HOME="$MX_DIR/cargo"
        if [[ ! -x "$CARGO_HOME/bin/cargo" ]]; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
        fi
        export PATH="$CARGO_HOME/bin:$PATH"
    fi
    if ! command -v protoc >/dev/null; then
        PROTOC_DIR="$MX_DIR/protoc"
        if [[ ! -x "$PROTOC_DIR/bin/protoc" ]]; then
            case "$(uname -m)" in
                x86_64) PROTOC_ARCH="x86_64" ;;
                aarch64) PROTOC_ARCH="aarch_64" ;;
                *) echo "Unsupported architecture $(uname -m)" >&2; exit 1 ;;
            esac
            curl --fail --location --silent --show-error \
                --output "$BUILD_DIR/protoc.zip" \
                "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-${PROTOC_ARCH}.zip"
            mkdir -p "$PROTOC_DIR"
            python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$BUILD_DIR/protoc.zip" "$PROTOC_DIR"
            chmod +x "$PROTOC_DIR/bin/protoc"
        fi
        export PATH="$PROTOC_DIR/bin:$PATH" PROTOC="$PROTOC_DIR/bin/protoc"
    fi
    git clone --quiet "$MODELEXPRESS_REPOSITORY" "$BUILD_DIR/modelexpress"
    (
        cd "$BUILD_DIR/modelexpress"
        git fetch --quiet origin "$MODELEXPRESS_REF"
        git checkout --quiet "$MODELEXPRESS_REF"
        cargo build --release --bin modelexpress-server
    )
    cp "$BUILD_DIR/modelexpress/target/release/modelexpress-server" "$BIN_DIR/"
    echo "$MODELEXPRESS_REF" > "$BIN_DIR/modelexpress-server.ref"
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
