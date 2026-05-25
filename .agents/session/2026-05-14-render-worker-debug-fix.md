# Session Handoff: BlenderGym Render Worker Debug & Fix

## 任务目的

修复 Render Service 的 Blender worker 子进程启动后 120s 超时崩溃的问题（`TimeoutError: Worker gpu=0 idx=0 not ready after 120s`），该问题阻塞了 8 卡 RL 训练。

## 问题诊断

在 debug pod 上逐步排查，发现了两个独立 bug：

### Bug 1: `ModuleNotFoundError: No module named 'blendergym'`

- **根因**: `worker_loop.py` 通过 `from blendergym.assets.pipeline_render_script import ...` 导入函数。这会触发 `blendergym/__init__.py`，拉起 `datasets`、`httpx`、`torch` 等重依赖。但 Blender 4.2 自带独立的 Python 3.11，没有这些包，且**忽略 `PYTHONPATH` 环境变量**（之前设置的 `PYTHONPATH` 完全无效）。
- **结果**: 每个 Blender worker 启动即崩溃，socket 永远不会 `bind`/`listen`，导致 `wait_ready()` 120s 超时。

### Bug 2: SIGPIPE (信号 13) 杀死 worker

- **根因**: `BlenderPool.wait_ready()` 通过 `connect()` + 立刻 `close()` 探测 socket 是否可连接。Worker 在 `accept()` 后尝试从已断开的连接 `recv` → `ConnectionError` → 异常处理器 `_send_json()` 向已关闭的 socket 写数据 → 产生 SIGPIPE。Blender 内嵌 Python **不抑制 SIGPIPE**（不同于标准 CPython），默认行为是终止进程。
- **结果**: 即使 Bug 1 修复后，worker 在 `wait_ready()` 探测后立刻死亡。

## 修复方案

### Fix 1: `blendergym/__init__.py` — PEP 562 lazy import

将顶层 import 改为 `__getattr__` 按需加载，`import blendergym` 不再触发重依赖链。Blender Python 做 `from blendergym.assets.pipeline_render_script import ...` 时只加载轻量的 `assets` 子包。

### Fix 2: `worker_loop.py` — `sys.path` + 正常 import

添加一行 `sys.path.insert(0, ...)` 把 blendergym 包根目录注入 Blender Python 的搜索路径，然后使用标准 `from blendergym.assets.pipeline_render_script import ...`。

### Fix 3: `worker_loop.py` — `signal.SIGPIPE` 忽略

在 `main()` 开头加 `signal.signal(signal.SIGPIPE, signal.SIG_IGN)`，抑制写入已断开 socket 时的 SIGPIPE 信号。这是长驻网络服务的标准做法。

## 修改的文件

| 文件 | 变更 |
|------|------|
| `blendergym/__init__.py` | 顶层 import → PEP 562 `__getattr__` lazy import |
| `blendergym/services/render/worker_loop.py` | `sys.path` 注入 + 正常 from import；`signal.SIGPIPE` 忽略 |
| `blendergym/services/render/persistent_blender.py` | 移除无效的 `PYTHONPATH` 和 `_blendergym_site_packages()` |

## 验证结果

- debug pod 上手动启动 Render Service（2 GPU）：health 返回 `"status": "ok"`，所有 worker `alive: true`
- 8 卡训练任务 `ericzyma-job-normal-20260514-175814` 成功启动并持续运行（截至 Step 10）：
  - Step 0: 1038.68s（含 OPTIX JIT 编译）
  - Step 1-3: ~100-110s/step，MFU 61-64%，~21k tokens/s
  - Step 4+: 部分 step 时间上升至 170-303s，MFU 下降至 37-42%
  - Peak Memory 从 77.5 GiB 逐步攀升至 115.4 GiB（Step 8 后稳定）
  - 无任何 "Render Service failed to start" 错误

**观察到的问题**：Step 4 起部分 step 耗时显著增加（300s vs 100s），Peak Memory 持续上涨（77→97→100→115 GiB），可能与 Blender 渲染 OOM 或显存碎片有关。

## 历史修复（同一 session）

- **Score Service GPU 争用**: CLIP 模型改为 lazy loading（首次 `/score` 请求时加载），避免 `check_gpus_available` 检测到 GPU 被占用
- **flashinfer 缺失**: 显式添加 `flashinfer-python` 和 `flashinfer-cubin` 到 `pyproject.toml` 依赖
- **Service Lifecycle Redesign**: `launcher.py`、`health.py`、sentinel 文件机制、`blendergym.sh` trap chaining 等（详见 `2026-05-14-blendergym-service-architecture.md`）

## 文档产出

- `.agents/kaola/troubleshooting.md` — 添加了 Blender worker 两个 bug 的详细记录
- `.agents/kaola/debugging.md` — **新建**，系统化的 env debug 方法论文档，包含：
  - 服务架构分层图（Mermaid）与日志位置速查
  - Debug Pod 工作流（热更新、editable install 验证）
  - 逐层隔离排查法（Layer 0-4）
  - 常用诊断命令
  - Blender 内嵌 Python 注意事项（PYTHONPATH、lazy import、SIGPIPE）
  - 典型失败模式速查表（9 种症状→原因→排查步骤）

## 下一步任务

1. **Blender OOM 调查**：Step 4+ 耗时飙升和 Peak Memory 持续攀升的根因分析
2. 观察训练是否会在更多 step 后触发 OOM crash
