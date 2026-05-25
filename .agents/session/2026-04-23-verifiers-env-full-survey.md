# Session Handoff: Verifiers 环境全面精读与知识更新

## 前置 session
- [VLM RL 探索与 VIGA 集成规划](2026-04-22-vlm-rl-exploration.md)
- [BlenderGym Verifiers 环境设计](2026-04-22-blendergym-env-design.md)
- [Verifiers 环境精读与知识整理](2026-04-23-verifiers-env-deep-read.md)

## 任务目的
在上一轮精读 SingleTurnEnv / MultiTurnEnv / Rubric 的基础上，系统浏览 verifiers 库的所有剩余环境类型（ToolEnv、StatefulToolEnv、SandboxEnv、PythonEnv、GymEnv、EnvGroup、BrowserEnv、OpenEnvEnv、MCPEnv、ReasoningGymEnv、TextArenaEnv），建立完整的环境继承体系认知，并更新知识文档。

## 执行内容
- 精读 `tool_env.py`：理解 ToolEnv 的 tool calling 模式——Python 函数自动转 JSON Schema、`env_response` 解析 tool_calls 并执行函数、`no_tools_called` 停止条件、`ToolMonitorRubric` 自动追踪调用指标
- 精读 `stateful_tool_env.py`：理解 `args_to_skip` 隐藏参数机制 + `update_tool_args` 运行时注入 + `filter_signature` 从 schema 中清除隐藏参数
- 精读 `sandbox_env.py`：理解 Docker 沙箱生命周期——`setup_state` 创建沙箱、`bash` tool 注册（args_to_skip=["sandbox_id", ...]）、`@vf.cleanup` 销毁沙箱、`@vf.teardown` 批量清理
- 精读 `python_env.py`：理解 FIFO 管道通信的持久化 Python REPL 架构
- 精读 `env_group.py`：理解多环境混合训练——`concatenate_datasets` 拼数据集、按 `task` 列路由 rollout/rubric、`EnvGroupRubric` 统一 metrics 命名空间
- 精读 `experimental/gym_env.py`：理解 Gym 适配模式——`gym_to_hf` 把 reset obs 转为 HF Dataset、`env_response` 中 action_parser/obs_to_text 双向转换、EpisodicSumRubric 累加每步 reward
- 浏览 `integrations/browser_env/`：BrowserEnv 继承 StatefulToolEnv，策略模式（DOM/CUA）通过 `_mode_impl.register_tools(self)` 动态注册 tools
- 浏览 `integrations/openenv_env.py`：OpenEnvEnv 继承 MultiTurnEnv，同时支持 gym 和 mcp 两种协议，每个 rollout 创建 Docker 沙箱 + expose 端口
- 浏览 `integrations/reasoninggym_env.py`：最简集成——reasoning_gym 数据集 + `score_answer` → SingleTurnEnv
- 浏览 `integrations/textarena_env.py`：Wordle 等文字游戏包装，deepcopy + shared_memo 优化，`final_env_response` 终止信号
- 浏览 `experimental/mcp_env.py`：MCPEnv 继承 ToolEnv，通过 stdio 连接 MCP server，MCPToolWrapper 把 MCP tool 包装成可调用对象
- 讨论知识文档更新方案：确定新增 ToolEnv / StatefulToolEnv / GymEnv / EnvGroup / 生命周期装饰器 / 继承关系全景图 + 决策指引
- 更新 `.agents/knowledge/env/verifiers_env_templates.md`：从 162 行扩展到 286 行

## 参考代码

### verifiers 库（uv cache 中，路径前缀 `~/.cache/uv/archive-v0/U894YVIYBpE9LizBTvajp/verifiers/`）
| 文件 | 说明 |
|------|------|
| `envs/tool_env.py` | ToolEnv + ToolMonitorRubric：无状态 tool calling |
| `envs/stateful_tool_env.py` | StatefulToolEnv：args_to_skip + update_tool_args |
| `envs/sandbox_env.py` | SandboxEnv：Docker 沙箱 + bash tool + cleanup/teardown |
| `envs/python_env.py` | PythonEnv：FIFO 管道 Python REPL |
| `envs/env_group.py` | EnvGroup + EnvGroupRubric：多环境混合训练 |
| `envs/experimental/gym_env.py` | GymEnv：Gym reset/step 适配器 |
| `envs/experimental/mcp_env.py` | MCPEnv：全局 MCP server 工具 |
| `envs/integrations/browser_env/browser_env.py` | BrowserEnv：DOM/CUA 双模式浏览器 |
| `envs/integrations/openenv_env.py` | OpenEnvEnv：OpenEnv gym/mcp 协议桥接 |
| `envs/integrations/reasoninggym_env.py` | ReasoningGymEnv：reasoning-gym 集成 |
| `envs/integrations/textarena_env.py` | TextArenaEnv：文字游戏集成 |
| `utils/tool_utils.py` | convert_func_to_tool_def：函数 → JSON Schema |

### 本次输出
| 文件 | 说明 |
|------|------|
| `.agents/knowledge/env/verifiers_env_templates.md` | 更新后的环境模板速查表（新增 6 个 section） |

## 最终方案
在原有 cheatsheet 基础上，按"升级路径"顺序插入 ToolEnv → StatefulToolEnv → GymEnv → EnvGroup → 生命周期装饰器的代码模板，末尾追加继承关系全景图和"该选哪个基类"决策表。集成类环境（BrowserEnv、OpenEnvEnv 等）只在全景图中一行带过，不单独展开模板，保持文档 cheatsheet 定位。

## 下一步任务
结合已有的 verifiers 环境全面认知，讨论 BlenderGym / BlenderBench 的技术选型——选择哪个基类、如何设计 reward、如何处理 Blender 渲染的部署架构。

## 初步方案

### 需要讨论的技术选型

1. **基类选择**：之前 session 倾向 MultiTurnEnv（多轮迭代），但现在完整了解了 GymEnv / StatefulToolEnv / SandboxEnv 后，需要重新评估：
   - **GymEnv**：BlenderGym 本身就有 reset/step 接口，GymEnv 可以直接适配——但 GymEnv 是 experimental 且不支持图片消息
   - **StatefulToolEnv**：模型通过 tool_call 提交 Blender 代码，env 在远端执行渲染——天然支持隐藏状态（Blender session），但需要 VLM 支持 tool calling
   - **MultiTurnEnv**（之前方案）：最通用，覆写 `get_prompt_messages` 每轮追加渲染图，不依赖 tool calling
   - **SandboxEnv**：如果 Blender 运行在 Docker 沙箱里，可以直接用——但 BlenderGym 可能不需要通用沙箱，而是需要专用 Blender 服务

2. **BlenderGym vs BlenderBench**：
   - BlenderGym：单步 3D 编辑，有 reset/step 接口，240 个任务
   - BlenderBench：更复杂的 benchmark，ref_based + ref_free 两阶段评估
   - 是先做 BlenderGym（简单验证链路），还是直接对标 BlenderBench？

3. **Reward 设计**：
   - 纯 CLIP similarity（baseline）
   - CLIP + photometric_loss 组合
   - CLIP + VLM judge（HybridRubric 模式）
   - 是否需要中间步骤的 reward（多轮场景下 reward shaping）

4. **渲染架构**：
   - Blender 进程管理：每个 rollout 启动新进程 vs 复用长驻进程
   - 与 env server 的通信方式：进程内调用 vs HTTP API vs ZMQ

### 建议准备
- 重新阅读 BlenderGym 的源码（`_reference_codes/VIGA/`），确认其 reset/step 接口的具体签名和返回值
- 对比 GymEnv 的 `StepResetEnv` Protocol 和 BlenderGym 的接口是否兼容
- 确认 BlenderBench 数据集的结构和评测流程
