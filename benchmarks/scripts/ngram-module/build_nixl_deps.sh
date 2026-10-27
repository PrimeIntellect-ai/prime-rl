#!/bin/bash
# Install NIXL weight-broadcast dependencies into the frozen arm worktrees (gitignored paths only).
# usage: build_nixl_deps.sh {ucx|nixl|modelexpress|link}; run on the GPU node.
# The node has no pkg-config, so NIXL gets the UCX path as a meson option or it silently skips its UCX plugin.
set -euo pipefail
MAIN=/home/garrett/github/PrimeIntellect-ai/prime-rl-run-ngram-main-522bdc0
BRANCH=/home/garrett/github/PrimeIntellect-ai/prime-rl-run-ngram-branch-4766d44
PREBUILT_UCX=/home/garrett/github/PrimeIntellect-ai/prime-rl-feat-ds-v4-nixl/third_party/ucx
UCX=$MAIN/third_party/ucx
WORK=/tmp/garrett/nixl-build
NIXL_VERSION=0.10.1
PROTOC_VERSION=28.3
mkdir -p "$WORK"

case $1 in
ucx)
    mkdir -p "$MAIN/third_party"
    cp -a "$PREBUILT_UCX" "$UCX"
    LD_LIBRARY_PATH="$UCX/lib:$UCX/lib/ucx" "$UCX/bin/ucx_info" -d | grep -E "Transport: (cuda_copy|rc_verbs)" | sort -u
    ;;
nixl)
    rm -rf "$WORK/nixl_source"
    git clone --depth 1 --branch "$NIXL_VERSION" https://github.com/ai-dynamo/nixl.git "$WORK/nixl_source"
    cd "$WORK/nixl_source"
    uv run --no-project --with tomlkit python contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml
    printf 'meson-python==0.21.1\n' > "$WORK/nixl-build-constraints.txt"
    uv venv --clear --python 3.12 "$WORK/build-venv"
    VIRTUAL_ENV="$WORK/build-venv" uv pip install pip
    rm -rf "$MAIN/third_party/nixl-wheels"
    mkdir -p "$MAIN/third_party/nixl-wheels"
    PATH=/usr/local/cuda-13/bin:$PATH CUDA_HOME=/usr/local/cuda-13 PKG_CONFIG_PATH="$UCX/lib/pkgconfig" \
        LD_LIBRARY_PATH="$UCX/lib:$UCX/lib/ucx" \
        "$WORK/build-venv/bin/python" -m pip wheel . --no-deps --wheel-dir="$MAIN/third_party/nixl-wheels" \
        --build-constraint "$WORK/nixl-build-constraints.txt" --config-settings=setup-args=-Ducx_path="$UCX"
    uv run --no-project python -m zipfile -l "$MAIN"/third_party/nixl-wheels/nixl_cu13-"$NIXL_VERSION"-*.whl \
        | grep libplugin_UCX
    for venv in /tmp/garrett/ngram-e2e2-main /tmp/garrett/ngram-e2e2-branch; do
        VIRTUAL_ENV=$venv uv pip install --reinstall --no-deps "$MAIN"/third_party/nixl-wheels/nixl_cu13-"$NIXL_VERSION"-*.whl
    done
    ;;
modelexpress)
    mkdir -p "$WORK/protoc"
    curl -fsSL -o "$WORK/protoc.zip" \
        "https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOC_VERSION/protoc-$PROTOC_VERSION-linux-x86_64.zip"
    uv run --no-project python -m zipfile -e "$WORK/protoc.zip" "$WORK/protoc"
    chmod +x "$WORK/protoc/bin/protoc"
    PATH="$WORK/protoc/bin:/home/garrett/.cargo-x86_64/bin:$PATH" PROTOC="$WORK/protoc/bin/protoc" \
        PROTOC_INCLUDE="$WORK/protoc/include" \
        bash "$MAIN/scripts/install_modelexpress.sh"
    ;;
link)
    mkdir -p "$BRANCH/third_party"
    ln -sfn "$UCX" "$BRANCH/third_party/ucx"
    ln -sfn "$MAIN/third_party/modelexpress" "$BRANCH/third_party/modelexpress"
    ls -la "$BRANCH/third_party"
    ;;
esac
