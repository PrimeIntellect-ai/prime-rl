# Session Handoff: VLM RL 探索与 VIGA 集成规划

## 任务目的
初学者视角了解 prime-rl 代码库，重点调研 VLM RL 训练能力，并规划将 VIGA (BlenderBench/BlenderGym) 集成为 verifiers 环境的路径。

## 执行内容
- 梳理了 prime-rl 整体架构：三大组件（Inference Server / Orchestrator / Trainer）及源码目录结构
- 确认了 VLM RL 支持现状：RL 训练 ✓（Qwen3-VL 系列），SFT ✗（SFT trainer 无多模态数据管线）
- 调研了 TRL 和 LLaMA-Factory 的 VLM 支持：TRL 有同步 VLM GRPO 但无异步；LLaMA-Factory VLM SFT 更成熟
- 对比了 TRL vs LLaMA-Factory 做 SFT 的优劣：LLaMA-Factory 在 SFT 效果、模板处理、VLM 模型覆盖上更优
- 分析了 prime-rl rollout 数据格式与 LLaMA-Factory 格式的差异：需要转换（消息结构扁平化、图片从 base64/file:// 转为路径、添加 `<image>` 占位符）
- 确认了 `teacher_rollout_model` 可对接闭源 API 收集数据（但无 logprobs，仅用于 SFT distillation，不能在线 RL）
- 初步浏览了 VIGA 代码库结构（evaluators、runners、agents、tools）

## 参考代码
| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/prime_rl/trainer/rl/data.py` | `TrainingSampleTensor` 类 | RL 训练数据结构，包含 `pixel_values`/`image_grid_thw`/`mm_token_type_ids` 多模态字段 |
| `src/prime_rl/trainer/rl/train.py` | L354-L371 | VLM 字段传入 forward，以及 VLM+CP 不兼容的检查 |
| `src/prime_rl/trainer/sft/train.py` | 全文件 | SFT trainer —— 无任何 VLM 支持 |
| `src/prime_rl/trainer/sft/data.py` | 全文件 | SFT 数据管线 —— 仅处理纯文本 |
| `src/prime_rl/orchestrator/trajectories.py` | `interleave_rollout()`, `VLMImageCache`, `offload_images_to_disk()` | VLM rollout 处理核心：图片缓存、base64→文件、interleave |
| `src/prime_rl/transport/types.py` | `TrainingSample` 类 | orchestrator→trainer 传输的数据结构定义 |
| `src/prime_rl/utils/vlm.py` | `VLM_REGISTRY` | VLM 模型注册表（qwen3_vl, qwen3_5, qwen3_5_moe, qwen3_vl_moe） |
| `src/prime_rl/configs/shared.py` | `ClientConfig` 类 | OAI 客户端配置，支持 `base_url`/`api_key_var`/`skip_model_check` |
| `src/prime_rl/configs/orchestrator.py` | `TeacherRolloutModelConfig` | 闭源 API rollout 生成配置 |
| `configs/multimodal/rl_color_codeword.toml` | 全文件 | 唯一的 VLM RL 配置示例（color-codeword 环境） |
| `docs/multimodal.md` | 全文件 | VLM 训练文档（限制、工作原理） |
| `_reference_codes/VIGA/` | 整个目录 | VIGA 参考代码，包含 BlenderBench/BlenderGym 的 evaluator 和 runner |
| `_reference_codes/VIGA/evaluators/blendergym/evaluate.py` | `clip_similarity()`, `photometric_loss()` | BlenderGym 评估指标：N-CLIP Score 和 Photometric Loss |
| `_reference_codes/VIGA/evaluators/blenderbench/evaluate.py` | 全文件 | BlenderBench 评估：ref_based + ref_free 两阶段 |
| `_reference_codes/VIGA/runners/blendergym/ours.py` | `load_blendergym_dataset()` | BlenderGym 数据加载和任务配置 |

## 最终方案
本次 session 是探索性的，未做代码修改。确定了推荐工作流：
1. 在 verifiers 中集成 BlenderBench/BlenderGym 环境
2. 用闭源 API（`teacher_rollout_model`）收集高质量 VLM rollout 数据
3. 筛选高 reward rollout，转换格式后用 LLaMA-Factory 做 VLM SFT（冷启动）
4. 用 SFT 模型部署 vLLM，再用 prime-rl 做 VLM RL 在线训练

## 下一步任务
将 `_reference_codes/VIGA` 中的 BlenderBench 和 BlenderGym 集成到 verifiers 环境中，先从简单的开始（BlenderGym 单步编辑）。

## 初步方案

### 入口与关键改动点

1. **先理解 verifiers 环境接口**：阅读 `verifiers` 库的环境定义规范（`vf.Environment` 基类），了解 `reset()`/`step()`/`score()` 需要返回什么结构（`TrajectoryStep`、`RolloutOutput`）

2. **从 BlenderGym 单步任务开始**（最简单）：
   - BlenderGym 是单步 3D 编辑（给一个起始场景 + 目标渲染图 → 模型输出 Blender Python 代码 → 渲染 → 评分）
   - 数据结构：每个任务有 `start.py`（起始代码）、`blender_file.blend`（场景文件）、`renders/start/`（起始渲染）、`renders/goal/`（目标渲染）
   - 任务类型：`geometry`(50)、`material`(40)、`blendshape`(75)、`placement`(40)、`lighting`(35)

3. **reward 函数设计**：
   - 复用 VIGA 的评估指标：`photometric_loss`（像素级 MSE）和 `clip_similarity`（语义相似度）
   - 简单方案：`reward = clip_similarity(render, goal)` 或组合指标
   - 注意：渲染需要 Blender 环境（headless），这是一个基础设施依赖

4. **VLM 特殊处理**：
   - prompt 中需要包含目标图片（作为 `image_url` 内容项）和可选的起始渲染图
   - verifiers 环境需要正确返回包含图片的 prompt 消息格式

5. **潜在风险**：
   - Blender headless 渲染的环境隔离和 GPU 占用
   - 渲染延迟可能成为瓶颈（每次 step 都需要调用 Blender）
   - BlenderGym 数据集需要单独下载（HuggingFace: `DietCoke4671/blenderbench`）
   - CLIP 模型加载的额外 GPU 显存开销
