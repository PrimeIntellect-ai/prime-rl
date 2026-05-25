# Session Handoff: Articraft 集成计划 v2 — 审查修正 + 最终定版

## 本次任务

在已有的三阶段集成计划（`articraft-env-integration_v2.md`）基础上，处理 Claude Code 审查反馈 + 自查发现的 10 处问题，全部修复后最终定版。

## 执行内容

### 第一批：自查发现的 8 处问题

| # | 问题 | 修正 |
|---|------|------|
| Fix 1 | Mermaid 图 `P1_tools` 残留（旧设计遗留） | 替换为 `P1_env` |
| Fix 2 | `mark_code_mutated()` 缺少 `MUTATING_TOOL_NAMES` 过滤 | 方法签名改为 `mark_code_mutated(self, tool_name: str)`，内部过滤 `frozenset({"apply_patch", "replace", "write_file"})`，9 处调用/描述全部更新 |
| Fix 3 | `build_compile_signal_bundle(report)` 调用错误 | 改为 `report.signal_bundle`（CompileReport 已含此属性），异常路径保留 `compile_signal_bundle_from_exception(exc)` |
| Fix 4 | `guidance.py` 在 Phase 1 脚手架中但功能属 Phase 2 | 从 Phase 1 脚手架移除，env_response 删除 guidance 调用，Diff 2b 标注为 Phase 2 Feature #2 |
| Fix 5 | `compile_urdf_report_maybe_timeout` 是同步函数 | 包 `asyncio.to_thread()` 防止阻塞 event loop |
| Fix 6 | `readable_paths` 概念不准确（articraft 无此概念） | 改为 `self.sdk_docs_context: str`（`load_sdk_docs_reference()` 返回值），VirtualWorkspace 保留在 rollout |
| Fix 7 | `TurnRecord` 未在 `env_response` 中构建 | 补充 TurnRecord 构建和 `rollout.turns.append()` |
| Fix 8 | Diff 1d 与 1.3 State 设计方案内容重复 | Diff 1d 改为引用 1.3，不再贴完整代码 |

### 第二批：Claude Code 审查反馈处理

审查报告来源：Claude Code 对 v2 计划的独立审计（~1000 行分析）。

**已确认无风险的项：**
- **红 #1**: `state["final_env_response"] = []` 终止 API — 经 verifiers 源码确认，是内置 `@vf.stop` 条件 `has_final_env_response`，完全支持
- **蓝 #6**: cadquery 用 `sdk_package` 字段过滤 — 经 articraft 源码确认不可行，所有 record 的 `sdk_package` 均为 `"sdk"`，正则是唯一方案

**采纳并修复的项：**

| # | 审查点 | 修正 |
|---|--------|------|
| 1 | `traj_id[:8]` 碰撞风险 | 改为 `[:12]`（3 处） |
| 2 | 安全阀终止时 reward 陈旧 | 加注释说明 rubric 已兜底（`bundle_dict is None → 0.0`） |
| 3 | compile timeout 默认 300s 太长 | TOML 配置表加 `URDF_COMPILE_TIMEOUT_SECONDS=30` |
| 4 | 缺少 observability metrics | Rollout 新增 `last_compile_latency_ms` 字段，rubric 新增 `compile_latency_ms` + `trajectory_token_estimate` 两个 metric |
| 5 | Checklist 不够全面 | 新增 3 条：vLLM 多行代码参数测试、compile latency 可见、token estimate 可见 |

**讨论后决定不改的项：**
- **Reward 设计**（`compile_attempted` 权重、build_failure 区间压缩）— Phase 1 先保留，训练后看数据再调
- **GuidanceInjector 维护债务** — 已 defer 到 Phase 2，届时加 `# SYNC:` 注释

## 关键调查结论

### verifiers 终止机制（红 #1 确认）

```
env_response() 设 state["final_env_response"] = []
  → 当轮跳过 get_model_response()
  → 下一轮 is_completed() 命中 has_final_env_response → 退出
  → render_completion() 构建 state["completion"]
```

`has_final_env_response` 是 MultiTurnEnv 内置的 `@vf.stop` 方法（priority=0），TextArena Wordle env 也使用同样模式。

### compile timeout 机制

`compile_urdf_report_maybe_timeout` 的 timeout 由环境变量 `URDF_COMPILE_TIMEOUT_SECONDS` 控制（默认 300s），函数本身不接受 timeout 参数。RL 训练中设为 30s。

### articraft 可读路径架构

articraft 没有 `readable_paths` 概念，实际是两条独立管线：
- `load_sdk_docs_reference(repo_root, sdk_package)` → SDK 文档字符串（给 Turn 0 prompt）
- `build_virtual_workspace(repo_root, model_file_path, sdk_package)` → VirtualWorkspace（给 read_file 工具）

SDK 文档加载不需要 `model_file_path`，可在 env `__init__` 调用一次共享。

## 产出文件

| 文件 | 说明 |
|------|------|
| `.agents/plans/articraft-env-integration_v2.md` | 最终定版（1834 行），含全部修正 |
| `.agents/plans/articraft-env-integration_v2.pdf` | PDF 导出版（Heiti SC 字体，中文正常） |

## 当前状态

**v2 计划已完成两轮审查修正（自查 8 项 + 外部审查 5 项），全部定稿。** 与 `2026-05-22-articraft-env-plan-finalized.md` 中的状态互补：前者记录三轮审计历史，本文件记录最终修正。

## 下一步

按 v2 计划 Phase 1 开始编码，优先级：
1. `schema.py` — Task/TurnRecord/Rollout 定义
2. `artifact_manager.py` — 路径管理 + 文件 I/O
3. `env.py` — ArticraftEnv 核心（setup_state + env_response + _dispatch_tool + _dispatch_compile）
4. `rubric.py` — Reward 计算 + metrics
5. `prompts.py` — System prompt + Turn 0
6. `dataset.py` — Records → HF Dataset
7. `test_rollout.py` — MockClient 端到端验证
