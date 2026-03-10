#!/bin/bash
# Builds the patched vllm-router (Rust) from our fork and installs it into .venv/bin.
# Requires: cargo (Rust toolchain)
set -euo pipefail

REPO_URL="https://github.com/S1ro1/router.git"
BRANCH="fix/preserve-extra-fields-disagg"
BUILD_DIR="/tmp/vllm-router-build"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="$SCRIPT_DIR/.venv/bin"

# Skip if already installed
if [ -x "$INSTALL_DIR/vllm-router-rs" ]; then
    echo "vllm-router-rs already installed, skipping."
    exit 0
fi

echo "Cloning $REPO_URL (branch: $BRANCH)..."
rm -rf "$BUILD_DIR"
git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$BUILD_DIR"

echo "Building vllm-router (release)..."
cd "$BUILD_DIR"
cargo build --release

echo "Installing to $INSTALL_DIR/vllm-router-rs..."
cp "$BUILD_DIR/target/release/vllm-router" "$INSTALL_DIR/vllm-router-rs"
chmod +x "$INSTALL_DIR/vllm-router-rs"

echo "Cleaning up..."
rm -rf "$BUILD_DIR"

echo "Done. Binary installed at: $INSTALL_DIR/vllm-router-rs"
