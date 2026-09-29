#!/usr/bin/env bash
# Install an optional MX client without changing the prepared GPU stack.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $(basename "$0") <modelexpress-client-source> [python]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLIENT_SOURCE="$(realpath "$1")"
# Keep the venv's interpreter path rather than resolving its system-Python symlink.
TARGET_PYTHON="$(realpath --no-symlinks "${2:-$PROJECT_DIR/.venv/bin/python}")"
RECORD_DIR="$PROJECT_DIR/third_party/modelexpress"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
mkdir -p "$RECORD_DIR"
cd "$PROJECT_DIR"

check_dependencies() {
    local phase="$1" status=0
    local receipt="$RECORD_DIR/dependency-check-$phase.log"
    uv pip check --color never --no-progress --python "$TARGET_PYTHON" > "$receipt" 2>&1 || status=$?
    cat "$receipt"
    if (( status > 1 )); then
        return "$status"
    fi

    # uv pip check reports conflicts from intentional dependency overrides.
    # Reject new diagnostics, unknown output and operational failures.
    awk -v status="$status" '
        /^Using Python .*environment at: / { next }
        /^Checked [0-9]+ packages? in / { next }
        /^All installed packages are compatible$/ { compatible = 1; next }
        /^Found [0-9]+ incompatibilit(y|ies)$/ { expected = $2; found = 1; next }
        /^The package `/ { print; count++; next }
        /^$/ { next }
        { invalid = 1 }
        END {
            if (invalid || (status == 0 && (!compatible || count)) ||
                (status == 1 && (!found || !count || count != expected))) exit 1
        }
    ' "$receipt" | LC_ALL=C sort -u > "$TEMP_DIR/$phase.txt" || {
        echo "Cannot interpret uv dependency check; see $receipt" >&2
        return 1
    }
}

check_dependencies before
uv pip freeze --python "$TARGET_PYTHON" --exclude-editable \
    | sed -E '/^(modelexpress|safetensors)([= @])/d' > "$TEMP_DIR/base-constraints.txt"
uv pip install --python "$TARGET_PYTHON" \
    --constraint "$TEMP_DIR/base-constraints.txt" \
    --reinstall-package modelexpress \
    safetensors==0.8.0 "$CLIENT_SOURCE"
check_dependencies after

LC_ALL=C comm -13 "$TEMP_DIR/before.txt" "$TEMP_DIR/after.txt" > "$RECORD_DIR/new-dependency-conflicts.txt"
if [[ -s "$RECORD_DIR/new-dependency-conflicts.txt" ]]; then
    echo "MX installation introduced dependency conflicts:" >&2
    cat "$RECORD_DIR/new-dependency-conflicts.txt" >&2
    exit 1
fi

uv pip freeze --python "$TARGET_PYTHON" > "$RECORD_DIR/requirements-installed.txt"
echo "MX client installed without introducing dependency conflicts"
