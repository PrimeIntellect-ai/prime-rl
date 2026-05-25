# Session Handoff: Articraft 环境集成 — Step 0 前期验证

## 任务目的

为 Articraft（3D 建模 agent）接入 prime-rl 做前期验证：审视集成方案、确定环境基类、基准测试 compile 性能、验证 KAOLA 集群上的依赖兼容性。

## 执行内容

1. **方案审视** — 对 `articraft-env-integration.md` 做全面 review，产出 12 条审视意见
2. **架构决策：MultiTurnEnv vs ToolEnv** — 深读 verifiers 源码（`MultiTurnEnv`/`ToolEnv`/`StatefulToolEnv`），确认 5 条 mismatch 理由，选定 `vf.MultiTurnEnv`
3. **compile_bench.py** — 编写并运行本地基准测试（20 条 records）：
   - 单次 compile 耗时：in-process 中位数 0.08s，subprocess 中位数 0.55s
   - 多 worker 并发：4→16 workers 实际并行度仅 2-3x（GIL + QC 锁瓶颈）
   - Outlier 问题：少数几何 QC 重的模型 compile 12-40s
4. **KAOLA 兼容性测试** — H200 debug pod 上实测 torch+vllm+articraft 共存：
   - 14/14 import 通过，CUDA matmul 正常
   - 安装方式：`uv pip install --no-deps -e articraft` + 手动装核心依赖（排除 cadquery）
5. **服务化评估** — 基于测试结果确认不需要服务化（无冲突 + compile 快 + 无资源隔离需求）
6. **verifiers API 验证** — 源码审查确认所有 API 假设：
   - `tool_defs` 通过 `Environment.__init__` kwargs 接受 ✅
   - `env_response` 返回 `[ToolMsg..., UserMsg?]` 兼容 TITO ✅
   - `state["final_env_response"]` 内置 stop condition ✅
   - `tool_call_parser` 对 Qwen3.5-9B 应为 `qwen3_coder` ✅
7. **计划持续更新** — 每轮决策后同步更新 `articraft-env-integration.md`

## 关键决策

| 决策 | 结论 | 理由 |
|------|------|------|
| 环境基类 | `vf.MultiTurnEnv` | ToolEnv 假设 1-tool-1-message、stateless、自动 JSON schema 注入，均不适用于 Articraft 的复杂 tool 语义（compile 返回结构化报告、tool 有副作用、需自定义终止逻辑） |
| 部署方案 | A'（直接安装，排除 cadquery） | KAOLA 实测零冲突；cadquery 是 lazy import 不在核心路径 |
| 服务化 | 不需要 | compile 中位数 0.08s，HTTP 开销是净负担；无 GPU 依赖，无资源隔离需求 |
| subprocess wrapper | RL 训练时建议禁用 | in-process 比 subprocess 快 7x（中位数）；用 max_turns 替代 compile 层超时 |
| tool_call_parser | `qwen3_coder`（Qwen3.5 系列） | `MODEL_TOOL_CALL_PARSER` 映射表确认；`hermes` 仅适用 Qwen3 |
| tool schema 注入 | 通过 env `__init__(tool_defs=...)` | verifiers 自动走 `state["tool_defs"]` → `get_model_response` → vLLM API |

## 产出文件

| 文件 | 说明 |
|------|------|
| `.agents/plans/articraft-env-integration.md` | 主计划文档，含 12 条审视意见 + Step 0 完整测试结果 + 结论摘要 |
| `environments/articraft/benchmarks/compile_bench.py` | compile 耗时基准测试脚本（单次 + 并发 + scaffold） |
| `environments/articraft/benchmarks/compat_test.py` | KAOLA 兼容性测试脚本（import + CUDA + 序列化 + compile） |
| `scripts/envs/articraft.sh` | KAOLA 环境 setup 脚本（apt install libfcl-dev + uv pip install） |

## 当前状态

**Step 0 全部完成（含 API 验证）。** 所有架构决策已确认并实测验证，verifiers/vLLM API 假设已通过源码审查确认，待进入 Step 1 正式实现。

剩余实现步骤：
- [ ] Step 1: 创建 `environments/articraft/` 包结构
- [ ] Step 2: 实现 `ArticraftEnv(vf.MultiTurnEnv)`
- [ ] Step 3: 实现 `ArticraftRubric(vf.Rubric)`
- [ ] Step 4: 实现 dataset loader
- [ ] Step 5: 编写 TOML 训练配置
- [ ] Step 7: 端到端测试

## 下一步建议

1. **Step 1-2 实现核心环境类** — 参考 `blendergym/env.py` 结构，重点实现 `env_response` 中的 tool 分发（compile_model 特殊处理 + 文件操作 tools）
2. **性能优化** — 训练时禁用 subprocess wrapper（`URDF_COMPILE_TIMEOUT_SECONDS=0`），可选关闭重 QC 作为 curriculum 初期简化
3. **SFT warm-up 数据** — 从 ~10K 成功 records 提取 (prompt, trajectory) pairs，9B 模型从零学 tool-use 3D 建模难度极大
