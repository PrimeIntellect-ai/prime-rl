#!/usr/bin/env bash
# Install the llm-d EPP (endpoint-picker) and an Envoy binary under
# third_party/llmd/bin/ for use by the `router_backend = "llm-d"` SLURM
# branches. No Docker, no sudo, no Kubernetes.
#
# EPP + pd-sidecar: built from source with `go install` against llm-d/llm-d-router.
# Envoy: downloaded directly from archive.tetratelabs.io (static tarball).
#   1.36+ is required — older Envoy lacks `FULL_DUPLEX_STREAMED` body mode
#   used by ext_proc.
#
# Usage:
#   bash scripts/install_llmd.sh                       # default versions
#   bash scripts/install_llmd.sh --epp-ref REV         # pin EPP git ref
#   bash scripts/install_llmd.sh --envoy-ver VER       # pin Envoy version (>=1.36)
#
# Requires: curl. Bootstraps Go locally if not on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Pin to a known-good revision rather than @latest. Bump as needed.
EPP_REPO="github.com/llm-d/llm-d-router"
EPP_REF="main"
ENVOY_VER="1.36.0"
GO_VER="1.23.4"

while [[ $# -gt 0 ]]; do
    case $1 in
        --epp-ref)   EPP_REF="$2";   shift 2 ;;
        --envoy-ver) ENVOY_VER="$2"; shift 2 ;;
        --go-ver)    GO_VER="$2";    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

DEST="$REPO_ROOT/third_party/llmd/bin"
GO_ROOT="$REPO_ROOT/third_party/llmd/go"
mkdir -p "$DEST"

command -v curl >/dev/null || { echo "ERROR: curl not on PATH" >&2; exit 1; }

if ! command -v go >/dev/null; then
    if [ ! -x "$GO_ROOT/bin/go" ]; then
        echo "=== Bootstrapping Go ${GO_VER} into ${GO_ROOT} ==="
        mkdir -p "$GO_ROOT"
        curl -fsSL "https://go.dev/dl/go${GO_VER}.linux-amd64.tar.gz" \
          | tar -xz -C "$GO_ROOT" --strip-components=1
    fi
    export PATH="$GO_ROOT/bin:$PATH"
fi
go version

echo "=== Building EPP + pd-sidecar from ${EPP_REPO}@${EPP_REF} ==="
TMP_GOBIN="$(mktemp -d)"
trap 'rm -rf "$TMP_GOBIN"' EXIT
GOBIN="$TMP_GOBIN" go install "${EPP_REPO}/cmd/epp@${EPP_REF}"
GOBIN="$TMP_GOBIN" go install "${EPP_REPO}/cmd/pd-sidecar@${EPP_REF}"
mv "$TMP_GOBIN/epp" "$DEST/epp"
mv "$TMP_GOBIN/pd-sidecar" "$DEST/pd-sidecar"
chmod +x "$DEST/epp" "$DEST/pd-sidecar"
echo "Installed: $DEST/epp, $DEST/pd-sidecar"

echo
echo "=== Fetching Envoy ${ENVOY_VER} from archive.tetratelabs.io ==="
TMP_ENVOY="$(mktemp -d)"
trap 'rm -rf "$TMP_GOBIN" "$TMP_ENVOY"' EXIT
curl -fL "https://archive.tetratelabs.io/envoy/download/v${ENVOY_VER}/envoy-v${ENVOY_VER}-linux-amd64.tar.xz" \
    -o "$TMP_ENVOY/envoy.tar.xz"
mkdir -p "$TMP_ENVOY/extract"
tar -xJf "$TMP_ENVOY/envoy.tar.xz" -C "$TMP_ENVOY/extract" --strip-components=1
cp "$TMP_ENVOY/extract/bin/envoy" "$DEST/envoy"
chmod +x "$DEST/envoy"
echo "Installed: $DEST/envoy"

echo
echo "Versions:"
"$DEST/epp" --version 2>&1 | head -1 || true
"$DEST/envoy" --version 2>&1 | head -1
echo
echo "Add to PATH (SLURM templates do this automatically):"
echo "  export PATH=\"$DEST:\$PATH\""
