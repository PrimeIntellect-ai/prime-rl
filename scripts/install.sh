#!/usr/bin/env bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

REPO_ID="prime-rl"

# Flag defaults (can be overridden via env)
SKIP_CLONE=${SKIP_CLONE:-0}

assert_supported_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  if [ "$os" != "Linux" ]; then
    log_error "Unsupported platform: ${os}/${arch}."
    log_error "prime-rl currently supports Linux on x86_64 or aarch64 only."
    exit 1
  fi

  case "$arch" in
  x86_64 | aarch64) ;;
  *)
    log_error "Unsupported architecture: ${arch}."
    log_error "prime-rl currently supports Linux on x86_64 or aarch64 only."
    exit 1
    ;;
  esac

  if ! command -v apt &>/dev/null; then
    log_error "Unsupported Linux distribution: apt is required by this installer."
    log_error "Use a Debian/Ubuntu-based Linux machine on x86_64 or aarch64, or install manually."
    exit 1
  fi
}

has_ssh_access() {
  # Probe GitHub SSH auth directly. `ssh -T git@github.com` returns
  # "successfully authenticated" only when the local key is registered
  # with a GitHub user, which is the precondition for cloning private
  # submodules over SSH. The previous `git ls-remote` probe could
  # spuriously succeed (cached creds, public-repo edge cases) and
  # leave submodule clones to fail later.
  set +e
  timeout 5s ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    -T git@github.com 2>&1 | grep -q "successfully authenticated"
  rc=$?
  set -e
  return $rc
}

ensure_known_hosts() {
  # Make sure ~/.ssh exists with the right perms, then add GitHub host key.
  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"
  # Use -H to hash hostnames; merge uniquely to avoid dupes.
  if command -v ssh-keyscan >/dev/null 2>&1; then
    ssh-keyscan -H github.com 2>/dev/null | sort -u |
      tee -a "${HOME}/.ssh/known_hosts" >/dev/null
    chmod 600 "${HOME}/.ssh/known_hosts"
  fi
}

# Initialize each submodule independently so that a missing private repo
# (e.g. configs/private when the user lacks access) does not abort the install.
init_submodules() {
  if [ ! -f .gitmodules ]; then
    return 0
  fi
  local paths failures
  paths=$(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')
  failures=()
  for path in $paths; do
    log_info "Initializing submodule: ${path}"
    if git submodule update --init --recursive -- "$path"; then
      :
    else
      log_warn "Could not initialize submodule '${path}' (likely no access). Continuing without it."
      failures+=("$path")
    fi
  done
  if [ "${#failures[@]}" -gt 0 ]; then
    log_warn "Skipped submodules: ${failures[*]}"
  fi
}

main() {
  assert_supported_platform

  # Ensure sudo exists
  if ! command -v sudo &>/dev/null; then
    apt update
    apt install -y sudo
  fi

  log_info "Installing base packages..."
  sudo apt update
  sudo apt install -y build-essential openssh-client curl git tmux htop nvtop

  log_info "Configuring SSH known_hosts for GitHub..."
  ensure_known_hosts

  if [ "$SKIP_CLONE" -eq 1 ]; then
    log_info "Skipping clone; assuming we are already inside the repo."
  else
    log_info "Determining best way to clone (SSH vs HTTPS)..."
    if has_ssh_access; then
      log_info "SSH access to GitHub works. Cloning via SSH."
      git clone git@github.com:PrimeIntellect-ai/${REPO_ID}.git
    else
      log_warn "SSH auth to GitHub not available. Cloning via HTTPS."
      git clone https://github.com/PrimeIntellect-ai/${REPO_ID}.git
    fi

    log_info "Entering project directory..."
    cd ${REPO_ID}
  fi

  if ! has_ssh_access; then
    git config url."https://github.com/".insteadOf "git@github.com:"
  fi

  log_info "Initializing submodules..."
  init_submodules

  log_info "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  log_info "Sourcing uv environment..."
  if ! command -v uv &>/dev/null; then
    source $HOME/.local/bin/env
  fi

  log_info "Installing prime..."
  uv tool install prime

  log_info "Syncing virtual environment..."
  uv sync --all-extras

  log_info "Installing pre-commit hooks..."
  uv run pre-commit install

  log_info "Installation completed!"
}

main
