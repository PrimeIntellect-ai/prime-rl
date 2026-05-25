# Articraft Environment Phase 1 — 实现 + 审视修复完成

**日期**: 2026-05-25
**状态**: Phase 1 代码完成 + 偏差审视修复完成，待 KAOLA 部署验证后 commit

---

## 完成的工作

基于 `articraft-env-integration_v2.md` 计划，完成了 Phase 1 全部代码实现（~1681 行）。
随后经 Claude Code 逐项审视偏差，修复了 2 项高/中优先级问题。

### 创建的文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `environments/articraft/articraft_env/schema.py` | 121 | Task/TurnRecord/Rollout dataclass，freshness 直接字段 |
| `environments/articraft/articraft_env/artifact_manager.py` | 189 | 路径管理 + meta.json/trajectory.json 写入 + 清理 |
| `environments/articraft/articraft_env/prompts.py` | 58 | 加载 system prompt + scaffold + Turn 0 消息构建 |
| `environments/articraft/articraft_env/dataset.py` | 154 | 扫描 records，过滤 cadquery，输出 HF Dataset |
| `environments/articraft/articraft_env/rubric.py` | 186 | 三维加权 reward + 5 metrics + @vf.cleanup |
| `environments/articraft/articraft_env/env.py` | 466 | ArticraftEnv 核心：setup/dispatch/compile |
| `environments/articraft/articraft_env/__init__.py` | 37 | PEP 562 lazy imports |
| `environments/articraft/pyproject.toml` | 19 | 包定义 |
| `environments/articraft/tests/test_rollout.py` | 257 | schema/reward/artifact 单元测试 |
| `configs/articraft/rl_articraft_kaola.toml` | 116 | KAOLA 训练配置 |
| `scripts/envs/articraft.sh` | 79 | 修改：补充 env 包安装 + URDF_COMPILE_TIMEOUT_SECONDS=30 |

### 修改的文件

- `scripts/envs/articraft.sh` — 新增 `setup_ac_install_env_pkg()` 和 `setup_ac_set_compile_timeout()`

---

## 审视后修复（高→低优先级）

详见 `.agents/plans/articraft-env-integration_v2-phase1-plan-vs-implementation-deviations.md`

| 偏差 | 风险 | 修复 |
|------|------|------|
| **#5** ToolResult content plain text vs JSON | 高 — 违反迁移一致性 | ✅ 改为 `json.dumps(result.to_dict() - tool_call_id)`，与 harness.py L888 一致 |
| **#2** system_prompt=None 失去 prefix caching | 中 — TITO 失效 | ✅ 改为传 `system_prompt=self.system_prompt_text`，setup_state 保留 system + 替换 user |
| **#9a** compile_required 文案 | 低 | ✅ 经对照确认与 harness_compile.py 原文**一致**，无需修改 |

### 保持不变的偏差

| # | 偏差 | 决策 |
|---|------|------|
| #1 | tool_defs 格式转换 | 必要且正确 |
| #3 | 双格式 accessor | KAOLA 后确认实际格式再精简 |
| #4 | dict messages | 与 BlenderGym 一致 |
| #6 | **kwargs 签名 | verifiers 推荐模式 |
| #7 | Path vs str | type-safe 改进 |
| #9c | TurnRecord 省略 arguments | 减少序列化开销 |

---

## 关键设计决策

1. **Rollout 直接字段替代 CompileFeedbackLoop** — `edit_revision`/`last_compile_revision` 直接在 Rollout dataclass 上，freshness 逻辑仅 3 行方法
2. **Tool dispatch 直接复用 articraft ToolRegistry/Invocation** — 零修改 articraft 代码
3. **tool_defs 格式转换** — `_convert_schemas_to_vf_tools()` 将 articraft OpenAI 格式转为 verifiers provider-agnostic 格式
4. **tool_call 双格式兼容** — accessor 函数同时处理 dict 和 Pydantic 对象（KAOLA 后精简）
5. **system_prompt 由 verifiers 管理** — 在 dataset 格式化阶段 prepend，setup_state 只覆盖 user messages
6. **ToolResult content JSON-wrapped** — `json.dumps(result.to_dict() - tool_call_id)`，与 harness.py 格式一致
7. **所有 prompt/文案与 articraft 原版完全一致** — 确保 RL 训出的模型迁移回 articraft 行为一致

---

## KAOLA 部署 Checklist

### 前置条件
- [ ] articraft repo 在 `/data/work/articraft`
- [ ] `scripts/envs/articraft.sh` 被 `setup_kaola.sh` source
- [ ] `URDF_COMPILE_TIMEOUT_SECONDS=30` 已 export

### 验证步骤（Step 1: 基础 smoke test）
- [ ] `uv pip install -e environments/articraft` 成功
- [ ] `python -c "from articraft_env import ArticraftEnv; print('OK')"` 成功
- [ ] `python -c "from articraft_env.dataset import build_dataset; ds = build_dataset('/data/work/articraft', split='train'); print(len(ds))"` — 预期 ~7000+ 条

### 验证步骤（Step 2: 单条 rollout）
- [ ] vLLM `tool_call_parser=qwen3_coder` + 4 个 tool schemas → 返回合法 tool_calls
- [ ] 完整 rollout 流程：setup → tool 执行 → compile → 终止 → reward
- [ ] reward 值域 [0, 1]，10+ 档连续值
- [ ] `completion_mask` 正确：assistant `[T...]` + env `[F...]` 交替

### 可能遇到的问题

1. **verifiers tool_defs 格式** — 如果 verifiers 不接受 dict 格式的 tool_defs（需要 `vf.Tool` Pydantic 对象），需修改 `_convert_schemas_to_vf_tools` 返回 `vf.Tool(...)` 实例
2. **env_response 消息格式** — verifiers 可能需要 Pydantic message 对象而非 dict；如果报错，需改为 `vf.ToolMessage(...)` / `vf.UserMessage(...)`
3. **compile 超时** — 如果 30s 太短导致大量 timeout，调整 `URDF_COMPILE_TIMEOUT_SECONDS`
4. **dataset 路径** — `data/records/` 在 KAOLA 上需确认存在且可读
5. **scaffold.py 位置** — 需确认 `/data/work/articraft/scaffold.py` 存在
6. **双格式 accessor** — KAOLA 验证时 log 实际收到的 tool_call 类型，确认后精简代码

---

## 同期其他变更（BlenderGym，同 repo 未 commit）

Claude Code 审视发现 repo 中还有未 commit 的 BlenderGym 改进：
- 服务启动架构重构（health.py + launcher.py + services.toml）
- Reward 权重可配置化
- Score Service lazy-load CLIP
- Blender worker SIGPIPE 修复 + sys.path.insert
- `__init__.py` lazy imports
- 3 个误 staged 的 .md 文件需 `git reset HEAD`

这些与 Articraft 独立，可分开 commit。

---

## 下一步

1. **KAOLA 验证**：按上述 checklist 逐步验证
2. **验证通过后 commit**：分组提交（Articraft Phase 1 / BlenderGym / 清理）
3. **Phase 1 完成后** → 进入 Phase 2（GuidanceInjector / repeated compile 检测 / failure_streak）
4. **Phase 2 按需启用** — 观察训练曲线后决定是否启用 guidance 等特性
