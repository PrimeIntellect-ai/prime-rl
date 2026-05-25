#!/bin/bash
# ============================================================================
# Articraft 环境插件 — 由 setup_kaola.sh 加载
# ============================================================================
#
# 这个脚本做什么：
#   在 KAOLA pod 上准备 Articraft RL 训练所需的一切：
#   代码、数据集、系统库、Python 包、环境变量。
#
# 被谁调用：
#   setup_kaola.sh 通过 `source scripts/envs/articraft.sh` 加载本文件，
#   然后调用 env_setup() 函数执行所有准备步骤。
#
# 可用的 base 变量（setup_kaola.sh source 时已定义）：
#   $S3_PREFIX    — S3 FUSE 挂载的用户根目录，如 /threed-code/ericzyma
#   $FAST_MODE    — "true" 时跳过非必要步骤（debug 快速启动用）
#   $PROJECT_DIR  — prime-rl 代码在容器内的路径，如 /data/work/prime-rl
#
# 函数命名约定：
#   所有函数用 setup_ac_ 前缀（Articraft 缩写），避免和 base 脚本或其他
#   env 插件（如 blendergym）的函数名冲突。
#
# ============================================================================

# --- 路径常量 ---
# ARTICRAFT_DIR: articraft 源代码在容器内的位置（agent/、sdk/、scaffold.py 等）
ARTICRAFT_DIR="/data/work/articraft"

# ARTICRAFT_CODE_TAR: 代码 tar 包在 S3 FUSE 上的路径（~16MB，含 sdk/agent/cli 等，不含 data/records）
ARTICRAFT_CODE_TAR="${S3_PREFIX}/data/articraft/articraft-code.tar"

# ARTICRAFT_DATASET_TAR: 数据集 tar 包在 S3 FUSE 上的路径（~1GB，含 10065 条 record）
ARTICRAFT_DATASET_TAR="${S3_PREFIX}/data/articraft/articraft-dataset-4-5star.tar"

# ARTICRAFT_DATASET_LOCAL: 数据集解压后的本地 SSD 路径（NVMe，读写快）
ARTICRAFT_DATASET_LOCAL="/local-ssd/data/articraft"


# --- 步骤 1: 解压 articraft 代码 ---
# 从 S3 FUSE 读取 tar 包，解压 articraft 源代码到容器本地盘。
# 为什么用 tar 而不是 aws s3 sync：articraft repo 去掉 data/records 后仍有 ~1400 文件，
# aws s3 sync 逐文件下载需要数分钟；tar 是单次顺序读 16MB 大文件，秒级完成。
# tar 包内容：sdk/、agent/、cli/、scaffold.py、pyproject.toml、data/categories/、
#             data/system_prompts/、data/batch_specs/（不含 data/records、data/cache、viewer/web）。
# 幂等：如果 sdk/ 目录已存在则跳过（pod 重启后重跑不会重复解压）。
setup_ac_sync_code() {
    echo "  [env] Extracting articraft code from tar..."
    if [ -d "${ARTICRAFT_DIR}/sdk" ]; then
        echo "    Already present, skipping"
        return
    fi
    if [ -f "${ARTICRAFT_CODE_TAR}" ]; then
        mkdir -p "${ARTICRAFT_DIR}"
        cat "${ARTICRAFT_CODE_TAR}" | tar xf - -C "${ARTICRAFT_DIR}/" --warning=no-unknown-keyword
        echo "    Extracted to ${ARTICRAFT_DIR} ($(find ${ARTICRAFT_DIR} -type f | wc -l) files)"
    else
        echo "    ERROR: code tar not found at ${ARTICRAFT_CODE_TAR}"
        exit 1
    fi
}


# --- 步骤 2: 解压数据集到本地 SSD ---
# 从 S3 FUSE 读取 tar 包，解压 10065 条 record 到本地 NVMe SSD。
# 为什么用 tar 管道而不是 cp -r：S3 FUSE 上 cp 大量小文件极慢（每文件一次 S3 API），
#   tar 管道是单次顺序读取一个大文件，实测快 30 倍（39s vs 20min）。
# 为什么放本地 SSD 而不是直接读 FUSE：训练时每个 rollout 都要读 record.json + model.py，
#   本地 SSD 随机读延迟 <1ms，FUSE 每次要走网络约 10-50ms。
# --strip-components=1: tar 包内顶层有一个目录（如 dataset/），解压时去掉它。
# 最后创建 symlink：让 articraft 代码树的 data/records/ 指向本地 SSD 上的数据，
#   这样 dataset.py 的 `root / "data" / "records"` 路径能找到数据。
# 幂等：如果 records/ 目录已存在则跳过。
setup_ac_restore_dataset() {
    echo "  [env] Restoring articraft dataset..."
    if [ -d "${ARTICRAFT_DATASET_LOCAL}/records" ]; then
        echo "    Already present, skipping"
    elif [ -f "${ARTICRAFT_DATASET_TAR}" ]; then
        mkdir -p "${ARTICRAFT_DATASET_LOCAL}"
        # cat + tar 管道：从 FUSE 顺序读 tar → 直接解压到本地 SSD
        cat "${ARTICRAFT_DATASET_TAR}" | tar xf - -C "${ARTICRAFT_DATASET_LOCAL}/" --strip-components=1 --warning=no-unknown-keyword
        echo "    Extracted to ${ARTICRAFT_DATASET_LOCAL} ($(ls ${ARTICRAFT_DATASET_LOCAL}/records/ | wc -l) records)"
    else
        echo "    WARNING: dataset tar not found at ${ARTICRAFT_DATASET_TAR}"
        echo "    Training will fail if no records are available."
    fi
    # 创建符号链接：让 /data/work/articraft/data/records → /local-ssd/data/articraft/records
    # -s: 创建符号链接（而非拷贝）
    # -f: 如果已存在则覆盖
    # -n: 如果目标是符号链接到目录，替换它而非在里面创建
    mkdir -p "${ARTICRAFT_DIR}/data"
    ln -sfn "${ARTICRAFT_DATASET_LOCAL}/records" "${ARTICRAFT_DIR}/data/records"
    echo "    Symlinked: ${ARTICRAFT_DIR}/data/records -> ${ARTICRAFT_DATASET_LOCAL}/records"
}


# --- 步骤 3: 安装系统级 C 库 ---
# libfcl-dev: FCL (Flexible Collision Library) 的开发头文件。
# python-fcl 包需要它来做碰撞检测（articraft QC 检查用）。
# apt-get update -qq: 静默更新包列表（-qq 减少输出）。
# --no-install-recommends: 不装推荐包（减小体积）。
# rm -rf /var/lib/apt/lists/*: 清理 apt 缓存（节省磁盘空间）。
setup_ac_install_system_libs() {
    echo "  [env] Installing system libraries (libfcl-dev)..."
    apt-get update -qq
    apt-get install -y --no-install-recommends libfcl-dev
    rm -rf /var/lib/apt/lists/*
}


# --- 步骤 4: 安装 articraft Python 包 ---
# 分两步安装：
#   第一步：--no-deps 安装 articraft 本身（只注册包，不拉依赖）。
#     为什么 --no-deps：articraft 的 pyproject.toml 声明了 cadquery（~200MB，我们不需要）
#     和 openai/anthropic/google-genai（LLM API 客户端，RL 训练不需要）。
#   第二步：手动安装我们真正需要的几何运算库。
#     manifold3d: CSG 布尔运算（构建 3D 几何体）
#     trimesh: 网格操作和 URDF 导出
#     python-fcl: 碰撞检测（QC 检查：零件重叠、零件隔离）
#     networkx: 图算法（铰接结构连通性检查）
#     scipy: 科学计算（几何变换等）
#     bm25s: BM25 文本检索（SDK 示例搜索）
#     zstandard: zstd 压缩解压（BM25 索引文件格式）
#     rich: 终端格式化输出
#   -e: editable install（开发模式，修改源码立即生效，无需重装）
#   --python: 指定目标 venv（KAOLA 镜像的 uv venv 在 /tmp/uv-venv）
setup_ac_install_python_pkg() {
    echo "  [env] Installing articraft SDK (no-deps, then core deps only)..."
    uv pip install --python /tmp/uv-venv/bin/python --no-deps -e "${ARTICRAFT_DIR}"
    uv pip install --python /tmp/uv-venv/bin/python \
        'manifold3d>=3.3.2' \
        'trimesh>=4.11.3' \
        'python-fcl>=0.7.0.8' \
        'networkx>=3.6.1' \
        'scipy>=1.17.1' \
        'numpy>=2.4.1' \
        'pydantic>=2.0.0' \
        'aiofiles>=24.1.0' \
        'rtree>=1.4.1' \
        'bm25s>=0.3.2.post1' \
        'zstandard>=0.23.0' \
        'rich>=13.7.0'
    echo "  [env] Articraft SDK installed (cadquery excluded)"
}


# --- 步骤 5: 安装 articraft-env 包 ---
# 这是我们写的 verifiers 环境包（environments/articraft/），
# 包含 ArticraftEnv、ArticraftRubric、dataset loader 等。
# 安装后 `from articraft_env import ...` 就能用了。
setup_ac_install_env_pkg() {
    echo "  [env] Installing articraft-env verifiers package..."
    uv pip install --python /tmp/uv-venv/bin/python \
        -e "${PROJECT_DIR}/environments/articraft"
    echo "  [env] articraft-env installed"
}


# --- 步骤 6: 设置编译超时 ---
# URDF_COMPILE_TIMEOUT_SECONDS: articraft compiler 用这个环境变量决定编译超时时间。
#   =0: 禁用子进程超时包装器，直接在当前进程编译（更快，但如果模型代码死循环会卡住）。
#   =30: 启用子进程包装器，30 秒内未完成则 kill（安全，但有 ~0.5s 额外开销）。
# RL 训练用 30s：防止个别 record 的编译死循环阻塞整个 worker。
# export: 让子进程也能读到这个变量（uv run python 启动的进程）。
setup_ac_set_compile_timeout() {
    export URDF_COMPILE_TIMEOUT_SECONDS=30
    echo "  [env] URDF_COMPILE_TIMEOUT_SECONDS=${URDF_COMPILE_TIMEOUT_SECONDS}"
}


# --- 步骤 7: 验证关键 import ---
# 确认所有核心库能正常导入且版本正确。
# 如果这步失败，说明依赖安装有问题，不应继续训练。
# 验证项：torch(GPU) + vllm(推理) + manifold3d/trimesh/fcl(几何) + sdk(articraft)
setup_ac_verify_imports() {
    echo "  [env] Verifying critical imports..."
    uv run python -c "
import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')
import vllm; print(f'  vllm={vllm.__version__}')
import manifold3d; print(f'  manifold3d OK')
import trimesh; print(f'  trimesh={trimesh.__version__}')
import fcl; print(f'  python-fcl OK')
import numpy; print(f'  numpy={numpy.__version__}')
import sdk; print(f'  articraft sdk OK')
print('  All imports successful!')
"
}


# --- 主入口 ---
# setup_kaola.sh 调用 env_setup()，按顺序执行上面所有步骤。
# 顺序有依赖关系：
#   sync_code 必须在 install_python_pkg 之前（install 需要 articraft 源码）
#   restore_dataset 可以和 install 并行，但为简单起见串行执行
#   verify_imports 必须最后（验证前面所有步骤的结果）
env_setup() {
    setup_ac_sync_code
    setup_ac_restore_dataset
    setup_ac_install_system_libs
    setup_ac_install_python_pkg
    setup_ac_install_env_pkg
    setup_ac_set_compile_timeout
    setup_ac_verify_imports
    echo "  [env] Articraft environment ready."
}
