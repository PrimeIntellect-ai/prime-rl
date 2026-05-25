# SETUP.md — 环境安装指南

## 目录总览

```
scripts/
├── install.sh                  # 本地开发一键安装（apt + clone + uv + pre-commit）
├── setup_kaola.sh              # KAOLA 集群训练环境编排器（通用主流程）
├── envs/
│   ├── blendergym.sh           # BlenderGym 环境插件（Blender + 渲染数据集）
│   └── articraft.sh            # Articraft 环境插件（SDK + 几何库 + 数据集）
├── install_deep_gemm.sh        # 可选：DeepGEMM FP8 推理加速（Hopper/Blackwell GPU）
├── install_ep_kernels.sh       # 可选：Expert Parallelism kernels（MoE 模型）
├── install_nixl_from_source.sh # 可选：NIXL 通信库（多节点权重同步）
├── fix-flash-attn-cute.sh      # 修复 flash-attn CUTLASS 编译问题
├── clean.sh                    # 清理训练产出（logs/checkpoints/wandb/rollouts）
├── tmux.sh                     # tmux 多窗口会话管理
├── chat.py                     # 交互式 chat 调试脚本
└── mini_moe.py                 # MoE 小规模测试脚本
```

---

## 1. 本地开发安装

```bash
# 方式 A：用 install.sh 一键完成（会 clone 仓库）
bash scripts/install.sh

# 方式 B：已有仓库，手动安装
uv sync --all-extras          # 安装所有依赖（含 flash-attn）
uv run pre-commit install     # 安装 git hooks
```

安装完成后验证：

```bash
uv run python -c "import prime_rl; print('OK')"
```

---

## 2. KAOLA 集群安装

KAOLA 上的安装由 `setup_kaola.sh` 编排，它是一个**通用主流程**，通过 `--env` 参数加载不同的环境插件。

### 2.1 整体流程

```
setup_kaola.sh
  │
  ├── [1/7] setup_python_deps     # uv sync --locked --extra flash-attn
  ├── [2-5/7] env_setup()         # ← 由 envs/<name>.sh 定义，做环境特有的安装
  ├── [6/7] setup_hf_cache        # 从 S3 tar 恢复 HuggingFace 模型缓存到本地 SSD
  └── [7/7] setup_s3_sync         # 启动后台进程，每 5 分钟同步训练产出到 S3
```

### 2.2 用法

```bash
# 在 koala submit 命令中（normal 模式，自动执行）：
koala submit -m normal -g 8 \
  --code "s3://arcwm-code-us-west-2/ericzyma/prime-rl:/data/work/prime-rl" \
  -c ". scripts/setup_kaola.sh --env articraft && uv run rl @ configs/articraft/rl_articraft_kaola.toml"

# 在 debug pod 中（交互式，手动 source）：
export EXP_NAME=my-experiment
export HF_TOKEN=hf_xxx
export WANDB_API_KEY=xxx
. scripts/setup_kaola.sh --env articraft --fast   # --fast 跳过数据集拷贝
```

### 2.3 参数

| 参数 | 说明 |
|------|------|
| `--env <name>` | 环境插件名，对应 `scripts/envs/<name>.sh`（默认 `blendergym`） |
| `--fast` | debug 模式：跳过数据集拷贝和 warmup，快速启动 |
| `--resume` | 跳过"S3 output 已存在"检查，用于从 checkpoint 恢复训练 |

### 2.4 必需的环境变量

| 变量 | 必须? | 说明 |
|------|-------|------|
| `EXP_NAME` | ✅ | 实验名称，决定 S3 输出路径和 checkpoint 目录 |
| `HF_TOKEN` | 推荐 | HuggingFace 认证，首次下载模型时需要 |
| `WANDB_API_KEY` | 推荐 | WandB 认证，记录训练 metrics |
| `HF_MODEL` | 可选 | 模型全称（默认 `Qwen/Qwen3.5-9B`），用于定位 HF cache tar |

---

## 3. 环境插件机制

每个环境插件是一个 bash 脚本，必须定义 `env_setup()` 函数。`setup_kaola.sh` 通过 `source` 加载它后调用 `env_setup()`。

### 3.1 Articraft 插件 (`envs/articraft.sh`)

做这些事（按顺序）：

| 步骤 | 函数 | 做什么 | 耗时 |
|------|------|--------|------|
| 1 | `setup_ac_sync_code` | 从 S3 下载 articraft 源码到 `/data/work/articraft` | ~10s |
| 2 | `setup_ac_restore_dataset` | 解压数据集 tar 到本地 SSD（10065 条 record） | ~30s |
| 3 | `setup_ac_install_system_libs` | apt install libfcl-dev（碰撞检测库） | ~5s |
| 4 | `setup_ac_install_python_pkg` | 安装 articraft SDK + 几何库（排除 cadquery） | ~20s |
| 5 | `setup_ac_install_env_pkg` | 安装 articraft-env verifiers 包 | ~3s |
| 6 | `setup_ac_set_compile_timeout` | 设置编译超时 30 秒 | 即时 |
| 7 | `setup_ac_verify_imports` | 验证所有关键 import | ~5s |

### 3.2 BlenderGym 插件 (`envs/blendergym.sh`)

做这些事：

| 步骤 | 做什么 | 耗时 |
|------|--------|------|
| 1 | apt install libegl1（OpenGL 渲染） | ~5s |
| 2 | 解压 Blender 4.2 到本地 SSD | ~8s |
| 3 | 解压渲染数据集（27GB） | ~39s |
| 4 | 安装 blendergym Python 包 | ~3s |
| 5 | 后台启动 OPTIX shader 编译预热 | ~6min（异步） |

### 3.3 写新插件

创建 `scripts/envs/my_env.sh`，实现 `env_setup()` 函数即可：

```bash
#!/bin/bash
# scripts/envs/my_env.sh

env_setup() {
    echo "  [env] Installing my environment..."
    # 你的安装逻辑
    echo "  [env] Done."
}
```

然后用 `--env my_env` 指定。

---

## 4. 可选组件

这些脚本只在特定场景需要，普通训练不用装。

### DeepGEMM（FP8 推理加速）

```bash
bash scripts/install_deep_gemm.sh
```

- **什么时候装**：用 FP8 量化模型做推理时（如 DeepSeek MoE）
- **要求**：CUDA 12.8+，Hopper 或 Blackwell GPU

### Expert Parallelism Kernels

```bash
bash scripts/install_ep_kernels.sh
```

- **什么时候装**：多节点 MoE 训练，需要 Expert Parallelism

### NIXL 通信库

```bash
bash scripts/install_nixl_from_source.sh
```

- **什么时候装**：多节点权重同步，替代默认的 NCCL broadcast

---

## 5. 其他脚本

### clean.sh — 清理训练产出

```bash
bash scripts/clean.sh
```

交互式确认后删除：`logs/`、`checkpoints/`、`weights/`、`rollouts/`、`wandb/`、`evals/`。

### fix-flash-attn-cute.sh — 修复 flash-attn

如果 `uv sync --extra flash-attn` 编译失败（CUTLASS 相关错误），运行此脚本。

---

## 6. 常见问题

详见：
- [`docs/troubleshooting.md`](../docs/troubleshooting.md) — prime-rl 通用问题（OOM、API 超时、TOML 解析等）
- [`.agents/kaola/troubleshooting.md`](../.agents/kaola/troubleshooting.md) — KAOLA 集群特有问题（S3 FUSE、pod 卡住、编码错误等）
