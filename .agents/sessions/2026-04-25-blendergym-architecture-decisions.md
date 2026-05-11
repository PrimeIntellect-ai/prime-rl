# Session Handoff: BlenderGym 环境架构决策

## 前置 session
- [VLM RL 探索与 VIGA 集成规划](2026-04-22-vlm-rl-exploration.md)
- [BlenderGym Verifiers 环境设计](2026-04-22-blendergym-env-design.md)
- [Verifiers 环境精读与知识整理](2026-04-23-verifiers-env-deep-read.md)
- [Verifiers 环境全面精读与知识更新](2026-04-23-verifiers-env-full-survey.md)

## 任务目的
在完整了解 verifiers 环境继承体系和 VIGA（BlenderGym/BlenderBench）代码的基础上，确定 BlenderGym 集成到 prime-rl 的完整架构方案——基类选型、模型选择、输出格式、Reward 设计、渲染部署、并发策略。

## 执行内容
- 深入分析 VIGA 的 BlenderGym 环境全流程：数据集结构（5 类 240 个任务）、dual-agent 架构（Generator + Verifier）、MCP tool server（exec.py 执行代码 + investigator.py 检查场景）、渲染管线（Blender subprocess + Cycles GPU 512×512）
- 深入分析 VIGA 的 BlenderBench 环境：3 级难度 27 个任务、ref-based（CLIP + photometric loss）+ ref-free（GPT-4o 4 维度打分）两阶段评测
- 对比四个候选基类（MultiTurnEnv / StatefulToolEnv / GymEnv / SandboxEnv），确定 **MultiTurnEnv** 为最终选择
- 讨论 Blender 有状态性（.blend 文件持久化）是否需要 StatefulToolEnv——结论：不需要，`state` dict + `env_response` 即可管理所有状态，StatefulToolEnv 解决的是"tool schema 参数隐藏"问题而非"环境有状态"问题
- 讨论文本反馈策略：VIGA 的 Verifier 文本反馈在 RL 训练中可被 scalar reward 部分替代，阶段 1 不加文本反馈，后续按需引入（CLIP 数值反馈 → VLM Verifier）
- 确定本地运行方案：不用 standalone env server，env worker 在本地 spawn Blender subprocess
- 确认 H20 GPU（96GB 显存）足以支撑多个并发 Blender 渲染进程
- 确定专用 1-2 张 H20 给环境交互 + verifier model，其余 GPU 用于 vLLM 推理和训练
- 确定使用 Qwen3.5-VL 模型、XML 输出格式（`<think>` + `<code>`）、sparse reward（最终轮 CLIP similarity）

## 参考代码

### VIGA 参考（`_reference_codes/VIGA/`）
| 文件 | 说明 |
|------|------|
| `main.py` | 入口：初始化 Generator/Verifier agent，运行主循环 |
| `agents/generator.py` | GeneratorAgent：多轮 tool_call 生成代码，memory 滑动窗口 |
| `agents/prompt_builder.py` | 构建 system prompt（初始图 + 目标图 + 初始代码） |
| `tools/blender/exec.py` | Blender Executor MCP Server：`execute_and_evaluate` tool，subprocess 调 Blender 渲染 |
| `tools/blender/investigator.py` | 3D Scene Investigation MCP Server：相机操作、场景查询 |
| `runners/blendergym/ours.py` | BlenderGym runner：加载数据集、并行执行任务 |
| `runners/blendergym/alchemy.py` | Alchemy 模式：best-of-N + VLM tournament 选择 |
| `evaluators/blendergym/evaluate.py` | 评测：CLIP similarity + photometric loss |
| `evaluators/blenderbench/ref_free_eval.py` | BlenderBench ref-free：GPT-4o 4 维度打分 |
| `prompts/blendergym/generator.py` | Generator system prompt 模板 |
| `prompts/blendergym/verifier.py` | Verifier system prompt 模板 |
| `prompts/blendergym/examples/` | 各任务类型的 few-shot 示例（placement/material/geometry/lighting/blendshape） |
| `data/blendergym/` | 数据集根目录，每个任务含 start.py + blender_file.blend + renders/ |
| `runners/shared/blender_executor.py` | Blender 代码执行工具函数 |

### prime-rl 侧
| 文件 | 说明 |
|------|------|
| `src/prime_rl/orchestrator/envs.py` | Env wrapper：spawn 本地 env server 或连接远程 |
| `src/prime_rl/orchestrator/env_server/env_server.py` | standalone env server 入口 |
| `configs/multimodal/rl_color_codeword.toml` | VLM RL 配置参考（Qwen3-VL-4B + color_codeword） |
| `docs/multimodal.md` | VLM 训练文档：VLMImageCache、MRoPE、限制事项 |

### verifiers 库（`~/.cache/uv/archive-v0/U894YVIYBpE9LizBTvajp/verifiers/`）
| 文件 | 说明 |
|------|------|
| `serve/server/zmq_env_server.py` | ZMQEnvServer：ZMQ ROUTER 前端 |
| `serve/server/env_router.py` | EnvRouter：worker pool，least-pending dispatch，worker 心跳重启 |
| `serve/server/env_worker.py` | EnvWorker：独立进程，加载 env 实例，async 处理 rollout |

### 知识文档
| 文件 | 说明 |
|------|------|
| `.agents/knowledge/env/verifiers_env_templates.md` | 环境模板速查表（含继承全景图 + 决策指引） |

## 最终方案

| 决策项 | 结论 | 理由 |
|--------|------|------|
| 基类 | **MultiTurnEnv** | 不依赖 VLM tool calling；灵活控制每轮图片消息；参考 color_codeword 已验证的路径 |
| 模型 | **Qwen3.5-VL** | 128K context 足以容纳多轮图片；prime-rl 已有 Qwen3.5 MoE VLM 支持 |
| 输出格式 | **XML**（`<think>` + `<code>`） | Qwen3.5 原生支持；verifiers 有 XMLParser |
| Reward | **最终轮 CLIP similarity**（sparse） | 简单起步，后续可扩展 HybridRubric |
| 文本反馈 | **阶段 1 不加** | RL 有 scalar reward 驱动学习，不像 VIGA 必须依赖文本反馈 |
| Memory 管理 | **抽象为钩子** | 在 `get_prompt_messages` 中实现，策略可配置 |
| Blender 运行 | **本地 subprocess** | 不用 standalone server，env worker 直接调 Blender |
| GPU 分配 | **1-2 张 H20 专用于 env**（Blender 渲染 + CLIP 评分 + 可选 VLM verifier） | H20 96GB 足以并发多个 Blender 进程 |
| Blender 安装 | **Infinigen 打包版** | 复用 VIGA 已有的 Blender 构建 |
| 数据集 | **下载 BlenderGym 原始数据**（240 个任务） | 5 类：blendshape(75) / geometry(50) / material(40) / placement(40) / lighting(35) |

### 为何不选其他基类

- **StatefulToolEnv**：其 `args_to_skip` + `update_tool_args` 解决的是"tool schema 参数隐藏"，不是"环境有状态"。Blender 状态通过 MultiTurnEnv 的 `state` dict 管理即可。且强依赖 VLM 支持 function calling 格式。
- **GymEnv**：experimental，`obs_to_text(obs) → str` 只支持文本 observation，无法返回图片。
- **SandboxEnv**：Blender 需要专用 GPU 渲染配置，不是通用 Docker bash 场景。过度抽象。

## 下一步任务
讨论并实现 BlenderGymEnv 的具体代码：env 文件结构、`setup_state`/`env_response`/`get_prompt_messages` 的具体逻辑、Rubric 实现、TOML 配置、数据集加载。

## 初步方案

### 需要实现的文件

1. **环境主文件**：BlenderGymEnv（继承 MultiTurnEnv）+ BlenderGymRubric（CLIP reward）+ `load_environment()` 工厂函数
   - `setup_state`：拷贝 .blend 文件到临时工作区、初始化 render 计数器、分配 GPU
   - `env_response`：解析 XML `<code>` → subprocess 调 Blender 渲染 → 返回渲染图
   - `get_prompt_messages`：第 0 轮构建 system prompt（初始图 + 目标图 + 初始代码 + 指令），后续轮拼接渲染结果
   - Rubric：最终轮的 CLIP similarity 作为 reward
   - `@vf.cleanup`：清理临时 .blend 和渲染目录

2. **Blender 执行工具**：`run_blender()` 函数，封装 subprocess 调用 + 渲染脚本 + 错误处理

3. **TOML 配置**：参考 `configs/multimodal/rl_color_codeword.toml`，设置 Qwen3.5-VL + blendergym env

4. **数据集加载**：扫描 BlenderGym 数据目录，构建 HF Dataset

### 关键设计点
- XML parser：`vf.XMLParser(["think", "code"], answer_field="code")`
- 图片消息格式：`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
- Blender 渲染脚本：复用 VIGA 的 `pipeline_render_script.py`（Cycles + CUDA + 512×512）
- 并发控制：通过 `num_workers` 配置 + 可选 asyncio.Semaphore 限制同时渲染数
- 视角选择：初期只用 render1（单视角），减少 token 消耗
