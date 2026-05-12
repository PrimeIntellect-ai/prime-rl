# Session Handoff: vLLM 升级与 Thinking Token Budget

## Rollout 失败问题统计（前序 session 分析结果）

对 `step_1/train_rollouts.jsonl` 的分析揭示了训练中的主要问题。以下是不同配置阶段的统计：

### 阶段 1：`max_completion_tokens=1024`（初始配置）

| 指标 | 值 |
|------|-----|
| 总 rollout 数 | 64 |
| 成功 | 33 (52%) |
| 失败 | 31 (48%) |

**失败分类**：
- **全部** 31 条失败都是 `xml_parse_success=0, render_success=0, reward=0`
- 没有 Blender 运行时错误 — 失败全部发生在 XMLParser 阶段（还没进入渲染）
- 20 条：最后输出有 `<code>` 但没闭合 `</code>`，代码写到一半被截断
- 11 条：整轮都在长篇分析（"I need to analyze..." / "Looking at the GOAL image..."），还没输出 `<code>` 就用完 token
- 大量失败样本 `output_tokens` 到了 `3072`（= 3 turns × 1024），完全耗尽 token budget

**按 example 聚合的失败率**：
```
example 8:  6/8 fail
example 20: 5/8 fail
example 38: 4/8 fail
example 26: 4/8 fail
example 37: 4/8 fail
example 23: 3/8 fail
example 25: 3/8 fail
example 41: 2/8 fail
```

### 阶段 2：`max_completion_tokens=4096`

| 指标 | step_0 | step_1 | step_2 |
|------|--------|--------|--------|
| 成功 | 54/64 (84%) | 59/64 (92%) | 58/64 (91%) |
| 失败 | 10 | 5 | 6 |
| 平均 output_tokens | ~5019 | ~5360 | ~4871 |

效果显著：失败率从 48% 降到 ~10%。剩余少量失败仍是 `<code>` 没闭合或没来得及输出 `<code>`。

### 阶段 3：`max_completion_tokens=8192` + `max_total_completion_tokens=12288`

在此配置下，剩余失败问题被识别为两大类（基于 conversation summary 中的数据）：
- **`xml_parse_failed` (~62%)**：模型在 `<think>` 阶段消耗过多 token，挤占实际代码输出空间
- **`GPU OOM` (~23%)**：Blender OPTIX 渲染时 GPU 显存不足（与推理共享 GPU）

`xml_parse_failed` 的根因进一步明确为 Qwen3.5 的 thinking mode：模型在 `<think>...</think>` 中进行大量推理，这些 token 被 `max_completion_tokens` 计入但最终被 chat template 从 response 中剥离，导致实际可用于代码输出的 token 大幅减少。

### 关键发现

1. **Thinking 是最大的 token 浪费源**：模型的 reasoning 消耗了大量 token budget，但这些内容在 chat template 处理后被剥离，不出现在最终的 assistant message content 中
2. **多轮累积 thinking 问题**：thinking 内容还会被保留在后续 turn 的 context 中（见下方详细分析），进一步挤压可用空间
3. **GPU OOM 是独立问题**：与 token budget 无关，需要单独解决（render 与推理共享 GPU）

---

## 本次完成的工作

### 1. 升级 vLLM 到 0.20.2+cu129

**问题**：训练 rollout 中主要失败为 `xml_parse_failed`，主因是 Qwen3.5-9B 在 `<think>` 阶段消耗过多 token，挤占了实际代码输出的空间。vLLM 0.20+ 支持 `thinking_token_budget` 参数可硬性限制 thinking token 数量。

**挑战**：直接 `pip install vllm>=0.20` 从 PyPI 拉到 CUDA 13.0 编译的 wheel，与 KAOLA 环境（CUDA 12.8）不兼容。级联导致 torchvision 也被升级到 CUDA 13.0 版本。

**解决方案**：
- 使用 vLLM GitHub Releases 上的 `cu129` 专用 wheel（兼容 CUDA 12.x PyTorch）
- 添加 `torchvision` 为直接依赖并指向 `pytorch-cu128` index（防止传递依赖拉到 CUDA 13 版本）
- vllm source 使用 `marker = "platform_machine == 'x86_64'"` 确保 aarch64 环境不受影响

**代码改动**：

`pyproject.toml`（3 处）：
```toml
# 1. 添加 torchvision 为直接依赖
dependencies = [
    ...
    "torchvision>=0.26.0",   # 新增
    ...
]

# 2. [tool.uv.sources] 添加 vllm cu129 wheel
vllm = [
    { url = "https://github.com/vllm-project/vllm/releases/download/v0.20.2/vllm-0.20.2+cu129-cp38-abi3-manylinux_2_31_x86_64.whl", marker = "platform_machine == 'x86_64'" },
]

# 3. [tool.uv.sources] 添加 torchvision 源
torchvision = { index = "pytorch-cu128" }
```

`configs/multimodal/rl_blendergym_kaola.toml`（2 处）：
```toml
[orchestrator.train.sampling]
max_completion_tokens = 8192
extra_body = { thinking_token_budget = 2048 }   # 新增

[inference.model]
max_model_len = 32768
enforce_eager = true
trust_remote_code = true
reasoning_parser = "qwen3"                       # 新增
```

**验证**：在 debug pod `ericzyma-job-debug-20260512-191916` 上确认：
- vLLM 0.20.2+cu129 正常 import
- torch 2.11.0+cu128 不受影响
- torchvision 0.26.0+cu128（修复了 CUDA 13 版本的 `torchvision::nms` 错误）
- `SamplingParams(thinking_token_budget=2048)` 正常工作

### 2. Flash Attention 调研结论

**Qwen3.5-9B 无法使用 FA3/FA4**：
- `model_type: qwen3_5`（不是 `qwen3_5_moe`），是 dense VLM
- 没有 custom implementation（`_CUSTOM_VLM_MAPPING` 只有 `qwen3_5_moe`）
- 走 HuggingFace 路径（`AutoModelForImageTextToText`）
- HF 路径只支持 `eager`、`sdpa`、`flash_attention_2`
- FA4 要求 `impl="custom"`；FA3 不被 HF 模型路径支持
- **FA2 已是当前默认且唯一可用选项**

### 3. 已提交的训练

Job: `ericzyma-job-normal-20260512-201055`
- 旧训练 `ericzyma-job-normal-20260512-174732` 已删除
- `s3/experiments/blendergym-9b-dp6` 已清空

---

## 下一步：多轮 Context 中 Thinking 内容的处理

### 发现的问题

通过代码追踪发现：**thinking 内容会被保留在后续 turn 的 context 中**。

完整数据流：

```
vLLM (reasoning_parser="qwen3")
  → 分离 content 和 reasoning_content
  → OpenAI API response: { content: "...", reasoning_content: "<think>...</think>" }

verifiers (openai_chat_completions_client.py)
  → parse_reasoning_content() 提取 reasoning_content
  → ResponseMessage(content=..., reasoning_content=...)

verifiers (response_utils.py: parse_response_message)
  → AssistantMessage(content=..., reasoning_content=..., ...)
  → 存入 trajectory step 的 "completion"

BlenderGymEnv.get_prompt_messages() (env.py L287-291)
  → 下一 turn: prev_prompt + prev_completion（包含 reasoning_content）

verifiers (openai_chat_completions_client.py: to_native_prompt)
  → ChatCompletionAssistantMessageParam(
        role="assistant",
        content=...,
        reasoning_content=message.reasoning_content,  # 传回 vLLM
    )

Qwen3 chat template
  → 将 reasoning_content 重新包装为 <think>...</think> 放入 tokenized prompt
```

### 影响

- **3-turn rollout** 最坏情况下 context 中有 ~4096 tokens 的 thinking 内容（2 个前置 turn × 2048 budget）
- 进一步挤压 `max_total_completion_tokens` 预算
- thinking 内容在后续 turn 中价值有限（模型已经看到渲染结果作为反馈）

### 拟议方案：清除前序 turn 的 reasoning_content

在 `BlenderGymEnv.get_prompt_messages` 中，构建后续 turn 的 prompt 时，清除 `prev_completion` 中 `AssistantMessage` 的 `reasoning_content`。

**不影响训练**：
- 每个 turn 的 `(prompt_ids, completion_ids, logprobs)` 仍然一致
- 模型看到什么 prompt → 生成什么 completion → logprobs 完全匹配
- 当前 turn 的 thinking tokens 仍然在 completion 中被训练

**正面效果**：
- 节省 context 空间（每个前序 turn 省 ~2048 tokens）
- 降低 `xml_parse_failed` 率
- 减轻 `max_total_completion_tokens` 压力

## 相关代码文件

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `environments/blendergym/blendergym/env.py` | L272-316 | `get_prompt_messages()` — **需要修改**，清除 `prev_completion` 中的 `reasoning_content` |
| `verifiers/.../response_utils.py` | `parse_response_message()` | 将 response 转为 `AssistantMessage`，保留 `reasoning_content` |
| `verifiers/.../openai_chat_completions_client.py` | L206 | `to_native_prompt()` — 将 `reasoning_content` 传回 API |
| `verifiers/.../openai_chat_completions_client.py` | L120-140 | `parse_reasoning_content()` — 从 response 提取 thinking |
| `verifiers/.../multiturn_env.py` | L120-143 | `add_model_response()` — 存储 completion 到 trajectory |
| `pyproject.toml` | L19, L127-130 | vLLM 和 torchvision 源配置 |
| `configs/multimodal/rl_blendergym_kaola.toml` | L51-53, L127 | thinking_token_budget 和 reasoning_parser 配置 |
| `src/prime_rl/configs/inference.py` | L90-95 | `reasoning_parser` 字段定义，传给 vLLM `--reasoning-parser` |
| `src/prime_rl/configs/orchestrator.py` | L116-121 | `extra_body` 字段定义，传给 vLLM sampling |

## verifiers 中的 Message 类型

verifiers 使用自定义消息类型（非标准 OpenAI SDK）：

```python
# verifiers/types.py (关键字段)
class AssistantMessage:
    content: str | list | None
    reasoning_content: str | None     # thinking 内容
    thinking_blocks: list | None      # Anthropic 风格的 thinking blocks
    tool_calls: list[ToolCall] | None
```

修改 `get_prompt_messages` 时需要构造新的 `AssistantMessage` 或深拷贝后清除 `reasoning_content`。注意 `prev_completion` 是 `list[AssistantMessage]`。

## 当前训练配置

```toml
seq_len = 16384
max_completion_tokens = 8192
max_total_completion_tokens = 12288
thinking_token_budget = 2048
reasoning_parser = "qwen3"
max_model_len = 32768
max_turns = 3
```

## 环境信息

- KAOLA 默认镜像：`cuda12.8-efa1.44-ubuntu24.04-zsh-uvcache`
- Debug pod：`ericzyma-job-debug-20260512-191916`（CUDA 12.8，已安装 vLLM 0.20.2+cu129）
- 当前训练 job：`ericzyma-job-normal-20260512-201055`
