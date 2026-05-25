# Session Handoff: Articraft 环境集成方案 — 计划定稿 + 三轮审计完成

## 任务目的

将 Articraft（3D 建模 agent）集成到 prime-rl 的完整设计方案经过三轮深度审计后定稿，所有致命/关键问题已在方案中修复，可直接进入 Step 1 编码。

## 执行内容

### 1. 三轮审计 → 修复全部致命问题

| 轮次 | 聚焦点 | 发现 |
|------|--------|------|
| 第一轮 | 源码级审计（50 条问题） | 识别 TOML 配置、reward 设计、token budget 等初步问题 |
| 第二轮 | 运行时流程一致性 | 发现 reward 二值 (FATAL)、compile 阻塞 event loop、readable_paths 并发、seq_len 不足 |
| 第三轮 | TITO/tool_call/reasoning 交互 | 确认 TITO 安全、malformed tool_call 处理正确、升级 thinking token 问题为致命 |

### 2. 已实施的关键修复（全部体现在方案中）

| # | 修复 | 影响 |
|---|------|------|
| 1 | `last_compile_attempt_dict` — 每次 compile 都存 bundle（含失败） | reward 从二值 → 10+ 档连续 |
| 2 | `asyncio.to_thread(compile_urdf_report_maybe_timeout, ...)` | 解除 worker event loop 阻塞 |
| 3 | `state["readable_paths"]` 替代 `self._readable_paths` | 多 rollout 并发安全 |
| 4 | Phase 1 禁用 `reasoning_parser` + `thinking_token_budget` | 避免 seq_len 被 thinking tokens 耗尽 |
| 5 | `seq_len = 32768` + Turn 0 docs 精简为 `DOCS_SUMMARY` (~800 tokens) | 给多轮对话留足空间 |
| 6 | Reward weights: `check_fraction=0.7, build_success=0.2, compile_attempted=0.1` | 总和 = 1.0，不超限 |
| 7 | 新增 `build_success_bonus` 函数 | 区分 build failure vs QC failure，GRPO 梯度更好 |
| 8 | `cadquery` 过滤改为 regex | 覆盖 `from cadquery import ...` 变体 |
| 9 | `tool_call.id` 空值 fallback | 防 vLLM 返回空 ID 时崩溃 |
| 10 | `last_compile_bundle_dict` 存 dict 而非对象 | JSON 序列化安全 |

### 3. 方案重组为三阶段渐进计划（v2）

产出 `articraft-env-integration_v2.md`（1834 行），结构：
- **Phase 1: Core In-Process RL** — env.py, rubric.py, schema.py, prompts.py, dataset.py, artifact_manager.py
- **Phase 2: Enhanced Feedback** — guidance injection, probe_model, find_examples, SFT warm-up, context window management
- **Phase 3: Visual Reward** — render service, CLIP scorer, rubric extension

### 4. 确认无问题的架构决策

- MultiTurnEnv + 手写 tool dispatch（不用 ToolEnv）
- `state["final_env_response"] = []` 终止机制
- `setup_state` 覆写 `state["prompt"]`（render_completion 切片一致）
- dataset `prompt=[]` + `system_prompt=None`（verifiers 正确透传）
- env_response 返回 `[ToolMsg..., UserMsg?]`（TITO 兼容）
- vLLM malformed output 自动降级为 text-only → freshness check
- ALL tool_call tokens 参与 loss（设计正确）

## 关键产出文件

| 文件 | 说明 |
|------|------|
| `.agents/plans/articraft-env-integration.md` | 完整方案（3852 行），含三轮审计修复、源码引用 |
| `.agents/plans/articraft-env-integration_v2.md` | 三阶段重组版（1834 行），模块化清晰 |
| `.agents/plans/articraft-env-integration_v2.pdf` | PDF 导出版（手机阅读用） |
| `environments/articraft/benchmarks/compile_bench.py` | compile 耗时基准测试脚本 |
| `scripts/envs/articraft.sh` | KAOLA 环境 setup 脚本 |

## 当前状态

**方案完全定稿，三轮审计全部通过，零残留问题。** 两个版本的计划文档互补：
- `articraft-env-integration.md`：详尽参考（含所有源码引用、边界情况分析、auditing history）
- `articraft-env-integration_v2.md`：行动导向（Phase 1/2/3 渐进，每个模块逐函数说明）

## Reward 最终设计

```
final_reward = 0.7 × compute_reward(bundle) + 0.2 × build_success + 0.1 × compile_attempted

compute_reward 值域 [0.0, 1.0]：
  - bundle is None → 0.0
  - SyntaxError → 0.05
  - RuntimeError → 0.10
  - 结构非法 → 0.15
  - Build 成功 + QC 失败 → 0.3 + 0.5 × (passed/total)
  - 全 QC 通过 + warnings → 0.8 + 0.1 × warning_clean_fraction
  - 全 QC 通过 + 无 warnings → 0.9 + 0.1 × efficiency
```

## 下一步（Step 1 编码）

1. 创建 `environments/articraft/articraft/` 包结构（7 个 Python 模块）
2. 逐模块实现 Phase 1 代码，参考 v2 计划中的逐函数规格
3. Mock eval 验证 token 用量 + reward 分布
4. TOML 配置 + KAOLA 端到端测试

## TOML 配置关键参数

```toml
seq_len = 32768
max_completion_tokens = 4096
max_total_completion_tokens = 8192
max_turns = 50
tool_call_parser = "qwen3_coder"
num_workers = 6
# Phase 1: 不启用 reasoning_parser
```
