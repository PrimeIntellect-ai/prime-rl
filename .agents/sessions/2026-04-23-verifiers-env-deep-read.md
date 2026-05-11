# Session Handoff: Verifiers 环境精读与知识整理

## 前置 session
- [VLM RL 探索与 VIGA 集成规划](2026-04-22-vlm-rl-exploration.md)
- [BlenderGym Verifiers 环境设计](2026-04-22-blendergym-env-design.md)

## 任务目的
精读 verifiers 库的三个参考环境（`reverse_text.py`、`color_codeword.py`、`math_env.py`）及其依赖的基类（`Environment`、`MultiTurnEnv`、`SingleTurnEnv`、`Rubric`、`JudgeRubric`），建立对 verifiers 环境接口的深度理解，并输出可供后续 Agent 直接使用的知识文档。

## 执行内容
- 精读 `reverse_text.py`：理解最小单轮环境范式（数据集 + parser + reward 函数 → SingleTurnEnv）
- 精读 `Environment` 基类和 `MultiTurnEnv`：理解 rollout 主循环（`init_state → while not is_completed: get_prompt_messages → get_model_response → add_model_response`）
- 精读 `color_codeword.py`：理解 VLM 多轮环境模式（覆盖 `get_prompt_messages` 每轮追加图片、`setup_state` 初始化状态、VLM image_url 消息格式）
- 追踪 Rubric 调用链：`run_rollout → rubric.score_rollout → _call_individual_reward_func`，理解参数注入机制（`inspect.signature` 按函数签名匹配 `prompt/completion/answer/state/task/info/parser`）
- 精读 `Rubric` 和 `JudgeRubric` 基类：理解 `add_reward_func` vs `add_metric`、`score_rollout` vs `score_group`、Judge 的 OpenAI API 调用
- 精读 `math_env.py`：理解 `HybridMathRubric`（规则验证 → LLM Judge 兜底 → 汇总）和 `python_tool=True` 时的 PythonEnv 模式
- 讨论知识文档的最佳形式：从"分 4 文件的概念文档"→"合并为一个面向 Copy-Paste 的 cheatsheet/template"
- 输出 `.agents/knowledge/env/verifiers_env_templates.md`：包含 SingleTurnEnv / MultiTurnEnv / Hybrid Rubric / VLM Image Message 模板、参数注入规则、TOML 配置连接方式

## 参考代码

### verifiers 库（uv cache 中）
| 文件 | 说明 |
|------|------|
| `~/.cache/uv/archive-v0/.../verifiers/envs/environment.py` | Environment 基类：init_state、run_rollout、run_group、generate |
| `~/.cache/uv/archive-v0/.../verifiers/envs/multiturn_env.py` | MultiTurnEnv：rollout 主循环、get_prompt_messages、env_response、stop 条件 |
| `~/.cache/uv/archive-v0/.../verifiers/envs/singleturn_env.py` | SingleTurnEnv：max_turns=1 的 MultiTurnEnv |
| `~/.cache/uv/archive-v0/.../verifiers/rubrics/rubric.py` | Rubric：score_rollout、_call_individual_reward_func（参数注入）、score_group |
| `~/.cache/uv/archive-v0/.../verifiers/rubrics/judge_rubric.py` | JudgeRubric：self.judge() 调用 OpenAI API 做 LLM Judge |
| `~/.cache/uv/archive-v0/.../reverse_text.py` | 最简单轮纯文本环境（48行） |
| `~/.cache/uv/archive-v0/.../color_codeword.py` | VLM 多轮环境，覆盖 get_prompt_messages 每轮加图片（285行） |
| `~/.cache/uv/archive-v0/.../math_env.py` | 双模式环境（SingleTurn / PythonEnv）+ HybridMathRubric（300行） |

### 本次输出
| 文件 | 说明 |
|------|------|
| `.agents/knowledge/env/verifiers_env_templates.md` | Agent 可用的环境模板速查表 |

## 最终方案
将 verifiers 环境知识整理为单文件 cheatsheet（`.agents/knowledge/env/verifiers_env_templates.md`），包含可直接套用的代码模板和关键注意事项，而非概念性文档。选择这种形式是因为：Agent 只需读一次文件即可拿到所有上下文，模板可直接 Copy-Paste 修改使用。

## 下一步任务
浏览 verifiers 库的其他 example 和源代码，找出本轮讨论未覆盖的系统和模式。

## 初步方案

### 已覆盖 vs 未覆盖

**已覆盖的环境基类**：`Environment`、`SingleTurnEnv`、`MultiTurnEnv`
**已覆盖的 Rubric**：`Rubric`、`JudgeRubric`
**已覆盖的示例**：`reverse_text`、`color_codeword`、`math_env`

**未覆盖的环境基类**（优先级高）：
- `tool_env.py` — Tool calling 模式，模型通过 function calling 调用工具
- `stateful_tool_env.py` — 有状态的 tool env
- `sandbox_env.py` — Docker 沙箱环境
- `env_group.py` — 多环境组合
- `python_env.py` — 仅浏览过，未深入（Docker 沙箱 Python REPL）

**未覆盖的集成环境**：
- `integrations/browser_env/` — 浏览器环境（CUA mode / DOM mode）
- `integrations/openenv_env.py` — OpenEnv 集成
- `integrations/reasoninggym_env.py` — ReasoningGym 集成
- `integrations/textarena_env.py` — TextArena 集成

**未覆盖的实验性环境**：
- `experimental/gym_env.py` — Gym 风格环境适配
- `experimental/mcp_env.py` — MCP 工具环境
- `experimental/harbor_env.py` — Harbor 任务环境
- `experimental/composable/` — 可组合环境系统（harness + taskset 模式）
- `experimental/cli_agent_env.py` / `opencode_env.py` / `rlm_env.py` — Agent 环境

**未覆盖的其他子系统**：
- `rubrics/math_rubric.py`、`rubrics/rubric_group.py` — 更多 Rubric 变体
- `clients/` — 多 provider 客户端（Anthropic、OpenAI completions 等）
- `serve/` — ZMQ env server 架构细节
- `rl/` — 内置 RL trainer
- `gepa/` — GEPA 系统
- `parsers/` — ThinkParser、MaybeThinkParser

**未覆盖的示例环境**（按兴趣排序）：
- `mmmu` — 多模态环境，可能与 BlenderGym 有参考价值
- `wiki_search` — 搜索工具环境
- `wordle` — 多轮猜词游戏
- `browser_cua_example` / `browser_dom_example` — 浏览器任务
- `tool_test` — 工具调用测试环境
- `self_reward` — 自我奖励环境
- `opencode_harbor` — 编程任务 + Docker 沙箱

### 建议浏览顺序
1. `tool_env.py` + `tool_test` 示例 → 理解 tool calling 模式
2. `env_group.py` → 多环境混合训练
3. `mmmu` → 多模态环境参考
4. `experimental/gym_env.py` → 看是否有 Gym 适配思路可用于 BlenderGym
5. `serve/` → env server 架构（BlenderGym standalone server 会用到）
6. 其余按兴趣探索
