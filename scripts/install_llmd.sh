#!/usr/bin/env bash
# Install the llm-d standalone (no-Kubernetes) routing binaries into
# third_party/llmd/bin:
#   - epp         : llm-d-router Endpoint Picker (the routing brain)
#   - pd-sidecar  : decode-side proxy for P/D disaggregation
#   - envoy       : the data-plane proxy that calls the EPP via ext_proc
#
# epp/pd-sidecar are built from a pinned llm-d-router commit. We currently build
# from a small fork that adds P/D disaggregation for vLLM's token-in
# /inference/v1/generate endpoint (prime-rl's renderer / TITO rollout path) —
# upstream's pd-sidecar only disaggregates the OpenAI endpoints, so token-in P/D
# silently runs decode-only. The fork is pending upstream PR
# llm-d/llm-d-router#1458; switch LLMD_ROUTER_REPO back to the upstream repo and
# bump LLMD_ROUTER_REF once it merges. The fork is branched off the upstream
# commit that added the EPP vllmhttp parser (PR #1248), which the renderer path
# also needs.
#
# System Go is not required: a Go toolchain is vendored under third_party/llmd/go
# and used to bootstrap; GOTOOLCHAIN=auto fetches the exact version the module
# pins. Envoy is extracted as a static binary from its release container image
# WITHOUT docker: we build `crane` (pure-Go, talks straight to the OCI registry)
# with the same vendored toolchain and use it to export the image filesystem.
# This is the only step that previously required a docker daemon, which cluster
# compute nodes don't have.
set -euo pipefail

LLMD_ROUTER_REPO="${LLMD_ROUTER_REPO:-https://github.com/S1ro1/llm-d-router.git}"
LLMD_ROUTER_REF="${LLMD_ROUTER_REF:-1ca4243ec84c657b4a5f507a1776d6c15a618d5b}"
GO_BOOTSTRAP_VERSION="${GO_BOOTSTRAP_VERSION:-1.23.4}"
ENVOY_VERSION="${ENVOY_VERSION:-1.36.0}"
ENVOY_IMAGE="${ENVOY_IMAGE:-envoyproxy/envoy:v${ENVOY_VERSION}}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
LLMD_DIR="$PROJECT_DIR/third_party/llmd"
BIN_DIR="$LLMD_DIR/bin"
GO_ROOT="$LLMD_DIR/go"
SRC_DIR="$LLMD_DIR/src"
GO_TOOLS_BIN="$LLMD_DIR/gotools/bin"
mkdir -p "$BIN_DIR" "$GO_TOOLS_BIN"

# --- Go toolchain (vendored bootstrap; auto-upgrades to the module's version) ---
if [ ! -x "$GO_ROOT/bin/go" ]; then
    echo "[install_llmd] downloading bootstrap Go $GO_BOOTSTRAP_VERSION"
    rm -rf "$GO_ROOT"
    curl -fsSL "https://go.dev/dl/go${GO_BOOTSTRAP_VERSION}.linux-amd64.tar.gz" | tar -xz -C "$LLMD_DIR"
fi
export GOROOT="$GO_ROOT"
export PATH="$GO_ROOT/bin:$PATH"
export GOTOOLCHAIN=auto
echo "[install_llmd] bootstrap $(go version)"

# --- Fetch llm-d-router source at the pinned ref ---
echo "[install_llmd] fetching ${LLMD_ROUTER_REPO}@${LLMD_ROUTER_REF}"
if [ ! -d "$SRC_DIR/.git" ]; then
    rm -rf "$SRC_DIR"
    git clone --quiet "$LLMD_ROUTER_REPO" "$SRC_DIR"
fi
git -C "$SRC_DIR" remote set-url origin "$LLMD_ROUTER_REPO"
git -C "$SRC_DIR" fetch --quiet origin
git -C "$SRC_DIR" checkout --quiet "$LLMD_ROUTER_REF"

# --- Build epp + pd-sidecar ---
echo "[install_llmd] building epp + pd-sidecar"
( cd "$SRC_DIR" && go build -o "$BIN_DIR/epp" ./cmd/epp && go build -o "$BIN_DIR/pd-sidecar" ./cmd/pd-sidecar )

# --- Envoy static binary (extract from the release image; keep if present) ---
# No docker: build crane with the vendored toolchain and export the image's
# filesystem straight from the registry, then pull out the static envoy binary.
if [ ! -x "$BIN_DIR/envoy" ]; then
    if [ ! -x "$GO_TOOLS_BIN/crane" ]; then
        echo "[install_llmd] building crane (OCI registry client, no docker needed)"
        GOBIN="$GO_TOOLS_BIN" GOFLAGS=-mod=mod \
            go install github.com/google/go-containerregistry/cmd/crane@latest
    fi
    echo "[install_llmd] extracting Envoy ${ENVOY_VERSION} from ${ENVOY_IMAGE}"
    "$GO_TOOLS_BIN/crane" export "$ENVOY_IMAGE" - \
        | tar -xf - -O usr/local/bin/envoy > "$BIN_DIR/envoy"
    chmod +x "$BIN_DIR/envoy"
fi

echo "[install_llmd] installed binaries in $BIN_DIR:"
"$BIN_DIR/epp" --version 2>&1 | head -1 || true
"$BIN_DIR/envoy" --version 2>&1 | head -1 || true
"$BIN_DIR/pd-sidecar" --help >/dev/null 2>&1 && echo "  pd-sidecar OK" || true
echo "[install_llmd] done"
