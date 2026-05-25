# Session Handoff: Articraft KAOLA 验证 → 训练提交 → Crash 修复全记录

## 前序 Session
- `.agents/session/2026-05-25-articraft-env-phase1-implementation.md` — Phase 1 代码实现完成
- `.agents/session/2026-05-22-articraft-env-plan-finalized.md` — 环境设计方案定稿
- `.agents/session/2026-05-25-blendergym-crash-analysis.md` — BlenderGym 4 天训练 crash 根因（S3 sync 失效 + thinking-only response）

## 任务目的

在 KAOLA 集群上完成 Articraft Phase 1 的完整验证，从单 GPU 功能测试到 8-GPU 正式训练启动。

---

## 阶段 1：KAOLA 功能验证（1 GPU debug pod）

### 执行内容

- 提交 1GPU debug pod，rsync articraft 代码到 pod
- 解压数据集 tar 到 `/local-ssd/data/articraft/records/`（10065 条 record）
- 安装依赖：prime-rl + articraft SDK(--no-deps) + 核心几何库 + articraft-env
- 验证全链路：async API ✅、tool dispatch ✅、dataset 构建（6898 train / 50 eval）✅、compile（225-1189ms）✅、reward 计算 ✅

### 调试经验

- **multiprocessing spawn**：测试脚本无 `__main__` guard 时 subprocess compile fork bomb，正式训练不受影响
- **cadquery 隐式依赖**：~1-2% record 通过了 import filter 但运行时依赖 cadquery → reward=0.10，不影响训练
- **bm25s 必装**：articraft import 链 `agent.tools.__init__` → `FindExamplesTool` → `bm25s`，遗漏导致整个 ToolRegistry 失败

---

## 阶段 2：训练提交调试（多次提交失败）

### 执行内容

- 修复 S3 sync：`setup_kaola.sh` 中 `rsync → S3 FUSE` 改为 `aws s3 sync` 直走 S3 API
- 修复 thinking-only response：env 的 `add_model_response` 中加 sanitize 代码
- 打包 articraft 代码 tar（13MB，排除 data/records + viewer/web 90k 无用文件）
- 修改 `articraft.sh` 用 tar 解压替代 aws s3 sync
- 多次提交遇到：guard check 误触发（FUSE 缓存幽灵目录）→ 加 `--resume`；坏节点 PodInitializing → 自动黑名单

### 调试经验

- **Mac tar xattr**：Mac 打的 tar 在 Linux 解压时 `set -e` 下 exit 1 → 打包加 `--no-xattrs --no-mac-metadata`，解压加 `--warning=no-unknown-keyword`
- **`set -euo pipefail` 传染**：source setup 脚本后主 shell 继承，debug 时需 `set +e`
- **aws s3 sync 92k 文件卡死**：articraft 逐文件下载太慢 → 改用 tar 单文件（秒级）

---

## 阶段 3：Orchestrator Crash 根因定位与修复（8 GPU）

### 问题 1：env_id 模块名冲突（根本原因）

**现象**：Orchestrator 启动 2 秒 exit code 1，`koala logs` 只显示错误码无详情。

**根因**：verifiers 的 `load_environment(env_id)` 内部直接 `importlib.import_module(env_id)`。config 中 `id = "articraft"` 导致导入 articraft **SDK** 包（同名顶层 module），而非 env 包 `articraft_env`。SDK 没有 `load_environment()` → AttributeError。

**修复**：`id = "articraft"` → `id = "articraft_env"`（train + eval 两处）

**教训**：verifiers 不用 entry_points，纯靠 module name 匹配。env 包的 importable name 必须唯一且与 config id 一致。

### 问题 2：WANDB_API_KEY 未注入 + shared mode 忽略 offline

**现象**：修复 env_id 后，orchestrator 和 trainer 都在 `wandb.init()` crash：`No API key configured`。

**根因**：
1. `WANDB_API_KEY` 不在 koala 自动注入列表（只注入 AWS 凭证）
2. `WANDB_MODE=offline` 无效：shared mode 下代码显式设 `Settings(mode="shared")` 完全覆盖环境变量

**修复**：提交命令用双引号展开本地 `$WANDB_API_KEY`：
```bash
-c "export WANDB_API_KEY=$WANDB_API_KEY ..."
```

### 问题 3：EXP_NAME 未设置（16 秒 instant fail）

**现象**：之前某次提交 16 秒失败。

**根因**：`-c` 用单引号，`$EXP_NAME` 未展开；或命令中遗漏 `export EXP_NAME=...`。

**修复**：在 `-c` 命令首部显式 `export EXP_NAME=articraft-9b-dp6`。

---

## 参考代码

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `configs/articraft/rl_articraft_kaola.toml` | L55-56, L77-78 | `id = "articraft_env"`（修复后） |
| `environments/articraft/articraft_env/__init__.py` | 全文 | lazy import，暴露 `load_environment` |
| `environments/articraft/pyproject.toml` | wheel packages | `["articraft_env"]` — importable name |
| `scripts/envs/articraft.sh` | 全文 | 完整 env 插件（tar 解压、依赖安装、验证） |
| `scripts/setup_kaola.sh` | L22, L74, L109-139 | `set -euo pipefail`、guard check、aws s3 sync |
| `src/prime_rl/utils/monitor/wandb.py` | L46-69 | shared mode 忽略 offline flag |
| `src/prime_rl/entrypoints/rl.py` | L160-220 | subprocess 启动 + env 传递 |
| verifiers `env_utils.py` (remote) | `load_environment()` | `importlib.import_module(env_id)` |

## 最终方案

正式训练提交命令（已验证成功启动）：
```bash
LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf-8 koala submit -m normal -g 8 \
  --code "s3://arcwm-code-us-west-2/ericzyma/prime-rl:/data/work/prime-rl" \
  -c "export EXP_NAME=articraft-9b-dp6 HF_MODEL=Qwen/Qwen3.5-9B WANDB_API_KEY=$WANDB_API_KEY && cd /data/work/prime-rl && . scripts/setup_kaola.sh --env articraft --resume && uv run rl @ configs/articraft/rl_articraft_kaola.toml"
```

## 当前运行状态

- **任务**: `ericzyma-job-normal-20260525-205047`（8 GPU，wandb 启用，正式训练）
- **状态**: Training loop started (step 0)，等待 eval rollout 完成后开始训练
- **Wandb**: `articraft-rl` 项目，run name `9b-dp6-bs64-kaola`
- **S3 日志**: `s3://arcwm-code-us-west-2/ericzyma/experiments/articraft-9b-dp6/output/logs/`

## 下一步任务

1. 确认首个训练 step 完成（reward > 0），wandb 出现 metrics
2. 打包 Qwen3.5-9B HF cache tar 到 S3（加速后续 pod 启动）
3. 根据训练曲线调参（reward weights、batch_size、max_turns）
4. Commit 所有改动（分组：env 代码 / config 修复 / 文档更新）
