#!/bin/bash
# Pod setup script for sweep-v2 experiments
# Run this ON a freshly provisioned Prime pod to prepare the environment.
set -euo pipefail

echo "=== Pod Setup: $(date) ==="

# --- Environment variables ---
export WANDB_API_KEY="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"
export PRIME_API_KEY="pit_f920c4b79c9c5568dda20eced3de43683dfdb6faea1e64063da80bad0cc0e781"
export PRIME_TEAM_ID="cmlr3u2er002zhr01tj8f48ts"

# Persist env vars for future shells
cat >> ~/.bashrc << 'ENVEOF'
export WANDB_API_KEY="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"
export PRIME_API_KEY="pit_f920c4b79c9c5568dda20eced3de43683dfdb6faea1e64063da80bad0cc0e781"
export PRIME_TEAM_ID="cmlr3u2er002zhr01tj8f48ts"
ENVEOF

# --- Install uv ---
echo "Installing uv..."
mkdir -p "$HOME/.config/fish/conf.d" 2>/dev/null || true
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "uv version: $(uv --version)"

# --- Clone prime-rl ---
REPO_DIR="$HOME/prime-rl"
BRANCH="worktree-sweep-eval-splits"

if [ -d "$REPO_DIR" ]; then
    echo "prime-rl already exists, pulling latest..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH" || true
else
    echo "Cloning prime-rl..."
    git clone --branch "$BRANCH" https://github.com/PrimeIntellect-ai/prime-rl.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# --- Set up submodules (clone at specific commits) ---
echo "Setting up submodules..."

setup_dep() {
    local dir="$1"
    local repo="$2"
    local commit="$3"
    if [ -d "$REPO_DIR/$dir/.git" ]; then
        echo "  $dir already cloned, checking out $commit..."
        cd "$REPO_DIR/$dir"
        git fetch origin
        git checkout "$commit"
    else
        echo "  Cloning $dir..."
        rm -rf "$REPO_DIR/$dir"
        git clone "$repo" "$REPO_DIR/$dir"
        cd "$REPO_DIR/$dir"
        git checkout "$commit"
    fi
}

setup_dep "deps/verifiers" \
    "https://github.com/PrimeIntellect-ai/verifiers.git" \
    "58b119fa1b24eff85b74a75ccf3e132523b3c6c3"

setup_dep "deps/renderers" \
    "https://github.com/PrimeIntellect-ai/renderers.git" \
    "8704f9d50252692a4a677177eb98d274f8d3ac5d"

setup_dep "deps/research-environments" \
    "https://github.com/PrimeIntellect-ai/research-environments.git" \
    "d141472268551411b6a9924a66b4426db3ce197d"

setup_dep "deps/pydantic-config" \
    "https://github.com/PrimeIntellect-ai/pydantic-config.git" \
    "94d71ecdec75cd4f6faf7ae4fe22b25e731a27e8"

# --- Install dependencies ---
echo "Running uv sync..."
cd "$REPO_DIR"
uv sync --all-extras 2>&1 | tail -5

# --- Validate ---
echo "Validating environment setup..."
uv run python3 -c "
from verifiers import load_environment
load_environment('reverse-text')
load_environment('wordle')
print('OK: reverse-text and wordle environments loaded successfully')
"

echo ""
echo "=== Pod Setup Complete: $(date) ==="
echo "Ready to run experiments from: $REPO_DIR/configs/sweep-v2/"
