#!/bin/bash
# ============================================================================
# KOALA 环境恢复脚本 — 通用编排器
# ============================================================================
# 用法：
#   . scripts/setup_kaola.sh [--fast] [--env blendergym]
#
# --fast   debug 模式（跳过数据集拷贝和 warmup）
# --env    环境插件名称（默认 blendergym），对应 scripts/envs/<name>.sh
#
# 注意：此脚本通过 source 执行（. scripts/setup_kaola.sh），set -euo pipefail
# 会影响调用方 shell。通过 koala submit -c 执行时无副作用（一次性 shell）；
# 在交互式 shell 中 source 时，后续命令也会受 set -e 约束。
#
# 环境变量（提交命令中 export）：
#   EXP_NAME      实验名称（必须设置，无默认值）
#   HF_MODEL      HuggingFace 模型全称（默认 Qwen/Qwen3.5-9B），用于 HF cache tar
#   HF_TOKEN      HuggingFace 认证
#   WANDB_API_KEY WandB 认证
# ============================================================================
set -euo pipefail

# --- 参数解析 ---
FAST_MODE=false
ENV_NAME="blendergym"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast) FAST_MODE=true; shift ;;
        --env)
            if [[ $# -lt 2 ]]; then echo "ERROR: --env requires a name"; exit 1; fi
            ENV_NAME="$2"; shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$FAST_MODE" = true ]; then
    echo ">>> Fast mode: skip dataset copy & warmup"
fi

# --- 环境变量 ---
export HF_HOME="/local-ssd/hf_cache"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set."
fi

# --- 路径配置 ---
if [ -z "${EXP_NAME:-}" ]; then
    echo "ERROR: EXP_NAME not set. Export it before running setup."
    echo "  e.g.: export EXP_NAME=blendergym-9b-dp6"
    exit 1
fi

HF_MODEL="${HF_MODEL:-Qwen/Qwen3.5-9B}"
HF_MODEL_SHORT=$(echo "${HF_MODEL}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')

S3_PREFIX="/threed-code/ericzyma"
S3_EXP="${S3_PREFIX}/experiments/${EXP_NAME}"
OUTPUT_LOCAL="/local-ssd/prime-rl-output"
CKPT_LOCAL="/local-ssd/checkpoints/${EXP_NAME}"
CKPT_S3="${S3_EXP}/checkpoints"
OUTPUT_S3="${S3_EXP}/output"
HF_CACHE_TAR="${S3_PREFIX}/tools/hf_cache_${HF_MODEL_SHORT}.tar"
PROJECT_DIR="/data/work/prime-rl"

# ============================================================================
# 通用函数定义
# ============================================================================

# 将 S3 上预打包的 HuggingFace 模型缓存解压到本地 SSD。
# tar 文件路径由 HF_MODEL 派生：Qwen/Qwen3.5-9B → hf_cache_qwen3.5-9b.tar
setup_hf_cache() {
    echo "  HF model cache (${HF_MODEL})..."
    if [ ! -d "${HF_HOME}/hub" ]; then
        if [ -f "${HF_CACHE_TAR}" ]; then
            cat "${HF_CACHE_TAR}" | tar xf - -C /local-ssd
            echo "    Restored from ${HF_CACHE_TAR}"
        else
            echo "    No tar at ${HF_CACHE_TAR}, will download on first use"
        fi
    else
        echo "    Already present, skipping"
    fi
}

# 安装 prime-rl 主框架的 Python 依赖（含 flash-attn 加速库）。
setup_python_deps() {
    echo "  Python dependencies..."
    uv sync --locked --extra flash-attn
}

# 启动后台进程，每 5 分钟将本地 SSD 上的训练产出同步到 S3（持久化）。
# shell EXIT 时触发最终同步，确保训练结束后不丢数据。
setup_s3_sync() {
    if [ "$FAST_MODE" = true ]; then
        echo "  Background sync: SKIPPED (fast mode)"
        return
    fi
    echo "  Starting background S3 sync..."
    mkdir -p "${CKPT_S3}" "${OUTPUT_S3}"
    sync_all() {
        # --inplace: 直接写入目标文件，跳过 rsync 默认的"临时文件 + rename"模式。
        # S3 FUSE 不支持 rename()（返回 ENOSYS），没有 --inplace 会导致同步静默失败。
        [ -d "${CKPT_LOCAL}" ] && rsync -a --inplace --delete "${CKPT_LOCAL}/" "${CKPT_S3}/" 2>/dev/null || true
        [ -d "${OUTPUT_LOCAL}" ] && rsync -a --inplace --copy-links --exclude broadcasts/ --exclude '*.bin' "${OUTPUT_LOCAL}/" "${OUTPUT_S3}/" 2>/dev/null || true
    }
    (while true; do sleep 300; sync_all; done) &
    SYNC_PID=$!
    echo "    PID: ${SYNC_PID} (every 5 min)"
    echo "    ${CKPT_LOCAL} -> ${CKPT_S3} (--delete)"
    echo "    ${OUTPUT_LOCAL} -> ${OUTPUT_S3} (--copy-links, excl broadcasts/*.bin)"
    trap "kill ${SYNC_PID} 2>/dev/null || true; [ -n \"\${OPTIX_PID:-}\" ] && kill \"\${OPTIX_PID}\" 2>/dev/null || true; sync_all" EXIT
}

# ============================================================================
# 加载 env 插件 + 执行主流程
# ============================================================================
cd "${PROJECT_DIR}"

ENV_SCRIPT="${PROJECT_DIR}/scripts/envs/${ENV_NAME}.sh"
if [ ! -f "${ENV_SCRIPT}" ]; then
    echo "ERROR: env script not found: ${ENV_SCRIPT}"
    exit 1
fi
source "${ENV_SCRIPT}"

# --- 主流程 ---
# 顺序：python deps → env_setup（OPTIX 后台启动）→ hf_cache / s3_sync（与 OPTIX 并行）→ wait
# env_setup 依赖 python deps（uv pip install 需要 venv），所以必须在其后。
# hf_cache 与 env_setup 无依赖，放后面可以和 OPTIX warmup 并行执行。
echo "=== [1/7] Python dependencies ==="
setup_python_deps

echo "=== [2/7~5/7] Environment: ${ENV_NAME} ==="
env_setup

echo "=== [6/7] HF model cache ==="
setup_hf_cache

echo "=== [7/7] Background S3 sync ==="
setup_s3_sync

echo "=== Setup complete ==="
# prime-rl 在训练启动前执行 check_gpus_available()，检测到 GPU 上有进程会拒绝启动。
# OPTIX warmup 使用 GPU 0 编译 shader，必须等它完成后再训练。
# wait 期间 OPTIX 已与 [6/7] hf_cache + [7/7] s3_sync 并行执行了一部分。
if [ -n "${OPTIX_PID:-}" ]; then
    echo "  Waiting for OPTIX warmup (PID: ${OPTIX_PID})..."
    wait "${OPTIX_PID}" 2>/dev/null || true
    echo "  OPTIX warmup done."
fi
echo "=== Ready to train. ==="
