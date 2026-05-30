#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "source this script: source scripts/env/activate-prime-rl.sh" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$repo_root/.venv" ]]; then
    deactivate 2>/dev/null || true
    unset VIRTUAL_ENV
fi

set -a
if [[ -f "$repo_root/.env" ]]; then
    # shellcheck disable=SC1091
    source "$repo_root/.env"
fi
set +a

# Slingshot/Cassini NCCL fabric (single source of truth; see scripts/env/isambard-fabric.sh)
# shellcheck disable=SC1091
source "$repo_root/scripts/env/isambard-fabric.sh"

export PRIME_RL_ROOT="$repo_root"
export PATH="$repo_root/.venv/bin:$PATH"

if [[ -f "$repo_root/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$repo_root/.venv/bin/activate"
else
    echo "No repo venv at $repo_root/.venv; run: env -u VIRTUAL_ENV uv sync --python 3.12 --all-extras" >&2
fi
