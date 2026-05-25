# Phase 1 计划 vs 实现：偏差记录

**日期**: 2026-05-25
**计划文件**: `.agents/plans/articraft-env-integration_v2.md`
**实现 commit**: Phase 1 全部代码

---

## 1. tool_defs 格式转换（计划未提及）

**计划**: 直接 `tool_defs=self.tool_registry.get_tool_schemas()` 传给 `super().__init__`。
**实现**: 新增 `_convert_schemas_to_vf_tools()` 函数，将 articraft 的 OpenAI legacy 格式 `{"type": "function", "function": {...}}` 转为 verifiers 的 provider-agnostic 格式 `{"name": ..., "description": ..., "parameters": ...}`。
**原因**: verifiers `Tool` 类型不接受 OpenAI wrapped 格式（探索 verifiers API 后发现）。
**风险**: 低。转换是无损的。但如果 verifiers 要求 `vf.Tool` Pydantic 对象而非 dict，还需进一步修改。

## 2. system_prompt 传递方式 ✅ 已修复

**计划**: 未明确说明 `system_prompt` 如何传给 `super().__init__`；伪代码中 `setup_state` 直接构建 `state["prompt"]` 含 system message。
**初始实现**: `super().__init__(system_prompt=None, ...)`，system message 完全由 `setup_state` 中手动构建到 `state["prompt"]` 里。
**修复后**: `super().__init__(system_prompt=self.system_prompt_text, ...)`，让 verifiers 在 dataset 格式化阶段（`_format_dataset` → `.map()`）自动 prepend system message。`setup_state` 保留已 prepend 的 system msgs，只替换 user portion。
**验证**: 确认了 verifiers rev 77a9f28 的调用链：`_format_dataset` prepend → `init_state` normalize → `setup_state` override user portion。

## 3. tool_call 访问格式兼容层（计划未提及）

**计划**: 伪代码使用 `tool_call.name` / `tool_call.arguments` 属性访问。
**实现**: 新增 `_extract_tool_calls` / `_tc_name` / `_tc_args` / `_tc_id` 四个 accessor 函数，同时兼容 dict 格式（`tc["function"]["name"]`）和 Pydantic 对象格式（`tc.function.name`）。
**原因**: verifiers 的 messages 在不同层级可能是 dict 或 Pydantic 对象。在 `env_response` 收到的 `messages[-1]` 的 `tool_calls` 结构不确定。
**风险**: 低。双格式兼容增加了鲁棒性，但增加了约 40 行代码。可能只有一种格式在实际运行中出现。

## 4. env_response 消息格式

**计划**: 返回 `[ToolMessage(role="tool", content=..., tool_call_id=...)]` 和 `[UserMessage(content="<compile_required>...")]`。
**实现**: 返回 plain dict `{"role": "tool", "content": ..., "tool_call_id": ...}` 和 `{"role": "user", "content": ...}`。
**原因**: verifiers 有 `maybe_normalize_messages()` 函数可以处理 dict 格式。BlenderGym 也使用 dict 格式。
**风险**: 中。verifiers 的 `_is_valid_env_tail()` 检测 TITO 兼容性时用 `_get_role(msg)` 检查 role，可能对 dict 和 Pydantic 对象都支持。但如果只支持 Pydantic 对象，需改为 `vf.ToolMessage(...)` / `vf.UserMessage(...)`。

## 5. ToolResult 内容提取方式 ✅ 已修复

**计划**: 使用 `result.to_content_str()` 方法构建 tool message content。
**初始实现**: 使用 `result.error if result.error else (result.output or "")` 直接提取，返回 plain text。
**修复后**: 改为 `json.dumps({k: v for k, v in result.to_dict().items() if k != "tool_call_id"})`，与 `harness.py` L888-895 完全一致。
**影响**: 保证 RL 训练和 articraft 推理时模型看到的 tool message content 格式一致（JSON wrapped），不影响迁移。

## 6. Rubric reward 函数签名

**计划**: `async def check_fraction_reward(self, state, info) -> float`。
**实现**: `async def check_fraction_reward(self, state: vf.State, **kwargs: Any) -> float`。
**原因**: verifiers 使用参数名注入（`inspect.signature()`），支持 `state`、`info`、`prompt`、`completion` 等。用 `**kwargs` 可以吸收所有未使用的注入参数，更安全。
**风险**: 低。verifiers 文档确认 `**kwargs` 是支持的。

## 7. compile_urdf_report_maybe_timeout 参数

**计划**: `compile_urdf_report_maybe_timeout(script_path=str(rollout.script_path), sdk_package=self.sdk_package)`，`script_path` 传 `str`。
**实现**: `compile_urdf_report_maybe_timeout(script_path=rollout.script_path, sdk_package=self.sdk_package)`，`script_path` 传 `Path` 对象。
**原因**: compiler.py 的签名是 `script_path: Path`，接受 Path 对象；内部 `_compile_worker` 会 `str(script_path)` 转换。
**风险**: 无。直接传 Path 更准确。

## 8. 行数估计偏差

| 文件 | 计划估计 | 实际行数 | 偏差原因 |
|------|---------|---------|---------|
| env.py | ~250 | 458 | 新增 tool_call accessor (~40行) + tool_defs 转换 (~20行) + 更完整的注释和错误处理 |
| rubric.py | ~100 | 186 | 新增 5 个 metric 函数（计划只提了 reward functions） |
| schema.py | ~60 | 121 | 与计划中 ~80 行的修订一致（含完整 Rollout 定义） |
| prompts.py | ~80 | 58 | 更简洁：直接复用 `agent.tools.build_first_turn_messages` |
| dataset.py | ~100 | 154 | record.json 解析逻辑更完整（revision 路径、prompt 回退逻辑） |
| artifact_manager.py | ~100 | 189 | 与 BlenderGym 模式更对齐（完整的 save_trajectory JSON schema） |
| test_rollout.py | ~80 | 257 | 更完整的单元测试（schema + reward + artifact 三组） |
| TOML | ~80 | 116 | 更完整（含 eval env 配置段） |
| 总计 | ~820 | 1674 | 约 2x |

## 9. 未实现的计划内容

### 9a. `compile_required` 提醒文案 ✅ 已确认一致

**计划**: 文案应与 `harness_compile.py L105-108` 完全一致。
**实现**: 与 `harness_compile.py` L105-108 原文**完全一致**，无需修改。
**验证**: 对照 harness_compile.py `append_compile_required_reminder()` 原文确认三行 XML 块一字不差。

### 9b. `_dispatch_compile` 成功时只调了 `mark_compile_success` 没有区分

**计划**: compile 成功时调 `mark_compile_attempt` + `mark_compile_success`；失败时只调 `mark_compile_attempt`。
**实现**: 一致。成功时两个都调，失败时只调 `mark_compile_attempt`。
**状态**: 符合计划。

### 9c. TurnRecord 中缺少 `arguments` 字段

**计划**: `tool_calls=[{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls]`。
**实现**: `tool_calls=[{"name": n} for n in tc_names]`，省略了 `arguments`。
**原因**: 减少序列化开销。arguments 可能很大（write_file 含完整代码）。
**风险**: 低。TurnRecord 主要用于 reward shaping 和 metrics，不需要完整 arguments。但调试时可能不便。

### 9d. `sdk._profiles.get_sdk_profile` 未直接使用

**计划**: 列为直接 import 的模块。
**实现**: 未直接 import — `get_sdk_profile` 在 `agent.workspace_docs` 内部使用，RL 层不需要直接调用。
**状态**: 不影响功能。plan 列出是说明依赖链，不要求 RL 层直接 import。

## 10. 计划中提及但实现中主动调整的设计

### 10a. `setup_state` 中的 prompt 构建 ✅ 已修复

**计划**: 
```python
state["prompt"] = build_turn0_messages(
    task.prompt_text,
    sdk_docs_context=self.sdk_docs_context,
)
```
**修复后**: 改为传 `system_prompt=self.system_prompt_text` 给 verifiers，`setup_state` 保留 verifiers prepend 的 system msgs，只替换 user portion：
```python
existing = state.get("prompt") or []
system_msgs = [m for m in existing if role(m) == "system"]
state["prompt"] = [*system_msgs, *build_turn0_messages(...)]
```
**差异**: 与计划更接近。verifiers 管理 system_prompt，setup_state 只覆盖 user messages。

### 10b. `build_turn0_messages` 多传了 `provider` 参数

**计划**: `build_turn0_messages(task.prompt_text, sdk_docs_context=self.sdk_docs_context)`
**实现**: `build_turn0_messages(task.prompt_text, sdk_docs_context=self.sdk_docs_context, provider=self.provider)`
**原因**: articraft 的 `build_first_turn_messages` 需要 `provider` 参数来选择 runtime guidance 文案（虽然当前所有 provider 返回同一文案）。
**状态**: 实现更准确。

---

---

## 修复记录（2026-05-25 审视后）

基于 Claude Code 逐项审视结果，执行了以下修复：

### 已修复

| # | 偏差 | 修复内容 |
|---|------|---------|
| #5 | ToolResult content plain text | 改为 `json.dumps({k:v for k,v in result.to_dict().items() if k != "tool_call_id"})`，与 harness.py L888-895 完全一致 |
| #2 | system_prompt=None | 改为 `system_prompt=self.system_prompt_text`，让 verifiers 在 dataset 格式化阶段管理 system message。`setup_state` 保留已 prepend 的 system msgs，只替换 user portion |
| #9a | compile_required 文案 | 经对照确认与 `harness_compile.py L105-108` **完全一致**，无需修改 |

### verifiers 调用链确认（rev 77a9f28）

- `system_prompt` 在 `_format_dataset` 阶段通过 `.map()` prepend 到每条 prompt（`environment.py L317-340`）
- `init_state` 只做 `normalize_messages`，不再 prepend system（`environment.py L558-621`）
- `setup_state` 在 `init_state` **之后**由 `MultiTurnEnv.rollout` 调用（`multiturn_env.py L146-177`）
- 因此：传 `system_prompt` → dataset 阶段自动 prepend → `setup_state` 中保留 system + 替换 user 部分

### 保持不变

| # | 偏差 | 原因 |
|---|------|------|
| #1 | tool_defs 格式转换 | 必要且正确 |
| #3 | 双格式 accessor | KAOLA 后根据实际格式精简 |
| #4 | dict messages | 与 BlenderGym 一致 |
| #6 | **kwargs 签名 | verifiers 推荐 |
| #7 | Path vs str | type-safe 改进 |
| #9c | 省略 arguments | 合理 trade-off |

---

## 总结

**核心逻辑符合计划**: freshness 机制、tool dispatch、compile 逻辑、reward 计算、artifact 管理均与计划一致。

**主要偏差集中在框架适配层**: tool_defs 格式转换、tool_call 双格式兼容 — 这些是基于 verifiers API 探索后的必要调整，计划没有预见到这些框架层面的细节。

**需要 KAOLA 验证的项目**:
1. tool_defs 是否需要 `vf.Tool` 对象而非 dict
2. env_response 返回的 dict 消息是否被 TITO 正确处理
3. tool message content 的 JSON wrapped 格式是否与 vLLM chat template 兼容
4. 双格式 accessor 实际只走 dict 还是 Pydantic 路径（确认后精简代码）
