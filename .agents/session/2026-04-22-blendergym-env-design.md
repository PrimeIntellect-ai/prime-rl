# Session Handoff: BlenderGym Verifiers 环境设计

## 前置 session
[VLM RL 探索与 VIGA 集成规划](2026-04-22-vlm-rl-exploration.md) — 梳理了 prime-rl 架构、VLM RL 支持现状、VIGA 代码结构，确定了"在 verifiers 中集成 BlenderGym"的方向。

## 本次 session 目的
从"怎么写一个 verifiers 环境"出发，深入研读 verifiers 接口和三个参考环境（`reverse_text.py`、`color_codeword.py`、`math_env.py`），设计 BlenderGym 环境的完整实施方案。

## 对话过程与关键决策

### 1. 理解 verifiers 环境接口
- verifiers 不是 Gym 风格的 `reset/step/reward`，而是"数据集行 -> init_state -> rollout -> rubric.score_rollout -> RolloutOutput"
- 核心抽象：`Environment` 基类、`MultiTurnEnv`（`env_response` + `get_prompt_messages`）、`SingleTurnEnv`（max_turns=1）
- 环境必须导出 `load_environment(**kwargs) -> vf.Environment` 工厂函数
- Rubric 的 reward func 自动注入 `prompt/completion/answer/state/task/info` 参数

### 2. 精读三个参考环境
- **`reverse_text.py`**（48行）：最小完整 env，纯文本单轮，不写子类直接实例化 `vf.SingleTurnEnv`
- **`color_codeword.py`**（285行）：VLM 多轮环境，图片 base64 data URL，覆盖 `get_prompt_messages` 每轮动态加图，独立 Rubric 类
- **`math_env.py`**（300行）：最复杂，`python_tool` 参数控制双模式（SingleTurnEnv / PythonEnv），HybridMathRubric 组合规则验证 + LLM judge

### 3. 初版 plan 与自我评审
- 初版方案：SingleTurnEnv 优先，ToolEnv 作为 Phase 2 扩展
- 自我评审指出的问题：
  - SingleTurnEnv 容易被误当成最终版（应明确为验证用途）
  - ToolEnv 有 VLM tool-calling 前提假设未验证
  - Blender 渲染不是辅助函数而是核心基础设施
  - CLIP reward 信号可能很弱（0.5-0.7 baseline）

### 4. 修订 plan：验证版 / 正式版分层
- Phase 1A：SingleTurnEnv 验证链路
- Phase 1B：MultiTurnEnv 正式版（参考 `color_codeword.py` 的 `get_prompt_messages` 模式，而非 ToolEnv）
- MultiTurnEnv 比 ToolEnv 更安全：不依赖 VLM tool-calling 支持，不依赖 tool response 里带图片

### 5. 部署架构讨论
- prime-rl 的 orchestrator 是 CPU 进程（Kubernetes 文档明确写 "No GPU required"）
- env worker 默认跑在 orchestrator 同节点（sidecar 模式），没有 GPU
- VIGA 不存在这个问题：它是串行分阶段的（API 生成 -> 本地 Blender 渲染 -> 离线 CLIP 评测），三者不并发
- Blender Cycles GPU 渲染 ~3-5s vs CPU ~30-60s，正式训练必须有 GPU

### 6. 最终决策：standalone env server + 独立 GPU（模式 B）
- env server 部署在有 GPU 的节点上，通过 `address` 配置连接
- 同一块 GPU 分时复用：Blender 渲染 -> CLIP 评分 -> VLM reward model（Phase 2）
- 显存预算：Blender ~2-4GB + CLIP ~0.6GB + 量化 VLM ~3-4GB = ~8GB 峰值
- 串行执行不会同时占用峰值
- CLIP 不强制 CPU，直接跑 env server 的 GPU（~10ms）
- 预计算 goal embedding 进一步减半 CLIP 推理量

### 7. Phase 2 扩展：VLM reward model
- 参考 `math_env.py` 的 `HybridMathRubric`：CLIP（weight=0.7）+ VLM judge（weight=0.3）组合评分
- VLM judge 可以是本地量化模型（Qwen3-VL-4B AWQ），也可以走外部 API（AsyncOpenAI client）
- 与 Blender/CLIP 共享 env server GPU

## 参考代码索引

### verifiers 环境（已安装在 .venv 中）
| 文件 | 说明 |
|------|------|
| `.venv/.../verifiers/envs/environment.py` | Environment 基类 |
| `.venv/.../verifiers/envs/multiturn_env.py` | MultiTurnEnv：env_response、get_prompt_messages、rollout 主循环 |
| `.venv/.../verifiers/envs/singleturn_env.py` | SingleTurnEnv：max_turns=1 |
| `.venv/.../verifiers/envs/tool_env.py` | ToolEnv：工具调用模式 |
| `.venv/.../verifiers/envs/python_env.py` | PythonEnv：Docker 沙箱 Python REPL |
| `.venv/.../verifiers/rubrics/rubric.py` | Rubric：score_rollout、score_group、add_reward_func/add_metric |
| `.venv/.../verifiers/types.py` | State、RolloutInput、RolloutOutput、TrajectoryStep 等核心类型 |
| `.venv/.../reverse_text.py` | 最简单的单轮纯文本环境 |
| `.venv/.../color_codeword.py` | VLM 多轮环境（本次最重要的参考） |
| `.venv/.../math_env.py` | 复杂双模式环境 + HybridMathRubric |

### prime-rl 侧
| 文件 | 说明 |
|------|------|
| `src/prime_rl/orchestrator/envs.py` | Env wrapper：run_rollout、run_group、ZMQ 连接 |
| `src/prime_rl/orchestrator/env_server/env_server.py` | standalone env server 入口 |
| `src/prime_rl/configs/orchestrator.py` | EnvConfig：address 字段、num_workers、args |
| `configs/multimodal/rl_color_codeword.toml` | VLM RL 配置参考 |

### VIGA 参考
| 文件 | 用途 |
|------|------|
| `_reference_codes/VIGA/runners/blendergym/ours.py` | 数据加载、blend 文件拷贝、GPU 分配 |
| `_reference_codes/VIGA/tools/blender/exec.py` | Blender 命令行调用、代码提取、渲染结果处理 |
| `_reference_codes/VIGA/data/blendergym/pipeline_render_script.py` | Cycles + CUDA + 512x512 渲染配置 |
| `_reference_codes/VIGA/evaluators/blendergym/evaluate.py` | clip_similarity、photometric_loss |

## 实施计划
完整 plan 见 `.cursor/plans/blendergym_verifiers_env_e0fae963.plan.md`，包含：
- 逐段伪代码 + 每段的参考来源
- 部署架构图（standalone env server + 独立 GPU）
- 系统风险清单
- 各阶段验收标准

## 下一步任务
按 plan 中的 TODO 顺序执行：
1. Phase 1A-infra：实现 `run_blender` + `clip_similarity`
2. Phase 1A：实现数据集构建 + SingleTurnEnv 验证版
3. Phase 1A-验收：跑通完整链路
4. Phase 1B：实现 `BlenderGymMultiTurnEnv`
5. Phase 1B-验收：多轮迭代跑通
6. Phase 2：加入 VLM reward model
