# Session Handoff: Context Length 与多模态截断问题

## 问题概述

BlenderGym RL 训练中，当 `max_completion_tokens` 较大（如 8192）时，多轮多模态序列的总长度可能超过 `seq_len`，触发 `batch.py` 中的截断逻辑。该截断逻辑对文本 token 正确处理，但**遗漏了 `pixel_values` 和 `image_grid_thw`**，导致图片 token 数量与图片 feature 数量不匹配，训练崩溃。

```
ValueError: Image features and image tokens do not match, tokens: 512, features: 576
```

## BlenderGym 多轮对话的图片结构

每个 rollout 的对话结构（`max_turns=3`）：

```
[System prompt]
[User Turn 1]: "GOAL image:" + IMAGE_1 + "INITIAL image:" + IMAGE_2 + code
[Assistant Turn 1]: reasoning + <code>...</code>
[User Turn 2]: "Render of your turn-1 program:" + IMAGE_3 + refine instruction
[Assistant Turn 2]: reasoning + <code>...</code>
[User Turn 3]: "Render of your turn-2 program:" + IMAGE_4 + refine instruction
[Assistant Turn 3]: reasoning + <code>...</code>
```

**图片累积规律**：
- Turn 1 prompt: 2 张图片（goal + initial）
- Turn 2 prompt: 3 张图片（goal + initial + turn-1 render）
- Turn 3 prompt: 4 张图片（goal + initial + turn-1 render + turn-2 render）

**每张图片的 token 数**：依据错误信息 `features: 576`，一张 256×256 图片约产生 576 个 vision token。

## Token 预算分析

以 Turn 3（最长情况）为例：

| 组成部分 | Token 数（估算） |
|---------|-----------------|
| System prompt | ~300 |
| Turn 1 prompt（text + 2 images） | ~300 text + 2×576 = ~1452 |
| Turn 1 completion（max） | max_completion_tokens |
| Turn 2 prompt（text + 1 image） | ~100 text + 576 = ~676 |
| Turn 2 completion（max） | max_completion_tokens |
| Turn 3 prompt（text + 1 image） | ~100 text + 576 = ~676 |
| Turn 3 completion（max） | max_completion_tokens |
| **合计** | **~3104 + 3 × max_completion_tokens** |

- `max_completion_tokens=1024` → 总长 ~6176（< seq_len=8192，安全）
- `max_completion_tokens=4096` → 总长 ~15392（< seq_len=16384，边界）
- `max_completion_tokens=8192` → 总长 ~27680（远超 seq_len=16384）

注：实际 completion 长度通常远小于 max。`max_completion_tokens=4096` 时实测平均 ~5000 总 token（3 turns 合计），最大 ~12288。

## Bug 定位

### 文件：`src/prime_rl/trainer/batch.py` L28-40

```python
if len(input_ids) > seq_len:
    input_ids = input_ids[:seq_len]
    loss_mask = loss_mask[:seq_len]
    inference_logprobs = inference_logprobs[:seq_len]
    position_ids = position_ids[:seq_len]
    advantages = advantages[:seq_len]
    temperatures = temperatures[:seq_len]
    if teacher_logprobs is not None:
        teacher_logprobs = teacher_logprobs[:seq_len]
    if routed_experts is not None:
        routed_experts = routed_experts[:seq_len]
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids[:seq_len]
```

截断了 `input_ids` 和 `mm_token_type_ids`（含图片占位 token），但 L76-78 直接透传：

```python
pixel_values=training_example.pixel_values,          # ← 未截断
pixel_values_shape=training_example.pixel_values_shape,  # ← 未截断
image_grid_thw=training_example.image_grid_thw,      # ← 未截断
```

### 错误触发条件

Qwen3-VL 的 forward 内部会对比：
- `mm_token_type_ids` 中 `==1`（image token）的数量
- `pixel_values` 中 vision encoder 输出的 feature 数量（由 `image_grid_thw` 决定）

当截断移除了部分 image token 但未移除对应 pixel_values → 不匹配 → ValueError。

## 数据结构关系

```
TrainingSample:
  prompt_ids: list[int]           # prompt token IDs（含 <|vision_start|>...<|vision_end|> 占位）
  completion_ids: list[int]       # completion token IDs
  mm_token_type_ids: list[int]    # 每个 token 的类型：0=text, 1=image, 2=video
  pixel_values: bytes             # 所有图片的 float32 feature 字节（已 flatten）
  pixel_values_shape: [N, D]      # N=total patches, D=patch_dim
  image_grid_thw: [[t,h,w], ...]  # 每张图片的网格尺寸，len = 图片数量

MicroBatch（prepare_sample 输出）:
  input_ids = prompt_ids + completion_ids（截断到 seq_len）
  mm_token_type_ids = 同步截断到 seq_len
  pixel_values = 透传（未截断）  ← BUG
  image_grid_thw = 透传（未截断）  ← BUG
```

### 图片 token 与 pixel_values 的映射

每张图片 i 在 `input_ids` 中占据连续的 image token（`input_ids[pos] == config.image_token_id`），在 `pixel_values` 中占据对应行数的 feature。`image_grid_thw[i] = [t_i, h_i, w_i]`。

**关键公式**：每张图片的 token 数 = `num_patches // spatial_merge_size²`，其中 `num_patches = t_i × h_i × w_i`。

**forward 中的配对机制**（`trainer/models/qwen3_5_moe/modeling_qwen3_5_moe.py` L815-822）：
```python
image_mask = (input_ids == self.config.image_token_id)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```
模型用 `masked_scatter` 把 vision encoder 输出填入 `image_token_id` 位置。如果截断删掉了部分 image token 但 `pixel_values` 仍为完整图像 patch，`masked_scatter` 会因 mask 元素数 != feature 数而报错。

**注意**：`mm_token_type_ids` 在 Qwen3_5MoeForCausalLM 的 forward 中**可能未被使用**（kwargs 未传入内部 VLM body）。实际配对完全依赖 `input_ids == image_token_id`。

### Packer 层的校验差异

- **`MultiPacker._validate_sample`**（`packer.py` L174-178）：要求 `prompt+completion ≤ seq_len`，超长直接 evict run
- **`SinglePacker`**：不做该校验，依赖 `prepare_sample()` 的硬截断

### `seq_len` / `max_model_len` / `max_completion_tokens` 无联合校验

框架中**不存在**同时校验这三个参数的 validator。仅有：
- `RLConfig.auto_setup_seq_len()`：确保 `trainer.model.seq_len ≥ orchestrator.seq_len`
- vLLM serving：确保 `prompt_tokens + max_tokens ≤ max_model_len`
- 三者的一致性需**人工保证**

## 可选解决方案

### 方案 A：正确截断 pixel_values（修复 batch.py）

在 `prepare_sample()` 中，当触发截断时：

1. 对截断后的 `mm_token_type_ids`，统计剩余的 image token 总数
2. 对比 `image_grid_thw` 中各图片的 patch 数，找到哪些图片完全保留、哪些被部分截断
3. 对于被部分截断的图片：
   - **选项 A1**：移除该图片的所有 token 和 pixel_values（保守，避免破损图片特征）
   - **选项 A2**：保留部分 token 但截断 pixel_values（不安全，vision encoder 可能无法处理不完整 patch）

```python
if len(input_ids) > seq_len and mm_token_type_ids is not None:
    # 统计截断后剩余的 image token（按图片边界对齐）
    truncated_mm = mm_token_type_ids[:seq_len]
    remaining_image_tokens = sum(1 for t in truncated_mm if t == 1)
    
    # 确定保留哪些完整图片
    kept_images = 0
    cumulative_tokens = 0
    for grid in training_example.image_grid_thw:
        img_tokens = grid[0] * grid[1] * grid[2]
        if cumulative_tokens + img_tokens <= remaining_image_tokens:
            cumulative_tokens += img_tokens
            kept_images += 1
        else:
            break
    
    # 只保留完整图片的 pixel_values
    # 同时把部分截断的图片 token 改回 text token (type=0)
    image_grid_thw = training_example.image_grid_thw[:kept_images]
    kept_patches = sum(g[0]*g[1]*g[2] for g in image_grid_thw)
    pixel_values = 截取前 kept_patches 行的 bytes
    # 修正 mm_token_type_ids：把不完整图片的 token 置 0
```

**优点**：从根本上解决问题，允许任意 seq_len/max_completion_tokens 组合
**缺点**：改动较复杂，需要处理 bytes 切片和边界情况

### 方案 B：提高 seq_len 避免截断

将 `seq_len` 设置足够大使截断永远不发生：

- `seq_len = 32768`（覆盖 3-turn 最坏情况 ~28K）
- H200 单卡 141GB，2 卡 DP 训练应该能承受

**优点**：零代码改动
**缺点**：
- 浪费显存（大部分样本远小于 32K）
- Trainer step 变慢（padding 到 32K 的样本仍消耗计算）
- 不从根本上解决问题（如果以后 max_turns 增加到 5+，又会超）

### 方案 C：优化 prompt + 限制 completion

1. 修改 `prompts.py` 去掉 "Optionally write a few sentences of reasoning"，要求直接输出 code
2. 保持 `max_completion_tokens=4096`、`seq_len=16384`（已验证 59/64 成功率）
3. 在 `prepare_sample()` 增加 warning log（而非崩溃）当截断影响图片时

**优点**：最小改动，已验证可行
**缺点**：不从根本上解决截断 bug，依赖"通常不会触发截断"的假设

### 方案 D：组合方案（推荐）

1. **修复 batch.py**（方案 A）— 确保截断时图片数据正确对齐
2. **保持 seq_len=16384** — 在 H200 上性能开销可接受
3. **max_completion_tokens=4096**（或 6144）— 平衡成功率和速度
4. （可选）**优化 prompt** — 减少不必要的 reasoning token

## 已有的 Context 管理机制

框架已有多个 context 相关设置，部分**尚未在 BlenderGym 配置中使用**：

### 1. `max_total_completion_tokens`（每个 env 级别）

**定义**：`configs/orchestrator.py` L317-325，`EnvConfig` 字段。

```toml
# 示例：限制 3 turns 总 completion 不超过 12288 tokens
[[orchestrator.train.env]]
max_total_completion_tokens = 12288
```

**工作原理**：
- 自动填入 `extra_env_kwargs` → 传给 `ZMQEnvServer` → `EnvWorker` 调用 `env.set_kwargs()` → 调用 `env.set_max_total_completion_tokens(value)`
- verifiers 的 `MultiTurnEnv` 有 `@vf.stop` 装饰的 `max_total_completion_tokens_reached()` 检查：当累计 `output_tokens >= max_total_completion_tokens` 时**提前终止 rollout**
- **当前 BlenderGym 配置中未设置**（默认 `-1` = 禁用）

**作用**：限制整个 rollout 的总生成 token 数，避免多轮累积失控。可与 per-turn 的 `max_completion_tokens` 协同使用。

### 2. `max_seq_len`（orchestrator → env）

**定义**：`configs/orchestrator.py` L1155。

自动将 `orchestrator.seq_len` 传给环境：
```python
env.extra_env_kwargs.update(max_seq_len=self.seq_len)
```

在 verifiers 的 `Environment` 基类中：
- `set_max_seq_len(value)` 存储为 `self.max_seq_len`
- 用于 `parse_response_tokens()` 中的 token 截断判断（`is_truncated`）
- 也用于 `OverlongPromptError` 的触发

### 3. `enable_prefix_caching`（推理侧）

**当前已启用**（`rl_blendergym_kaola.toml` L119）。vLLM 的 prefix caching 对多轮对话有性能帮助：后续 turn 的 prompt 与前一 turn 共享大量前缀，KV cache 可复用，跳过重复 prefill。

### 4. `OverlongPromptError`（verifiers 层）

当某一轮的 prompt 长度超过 `max_model_len` 时，vLLM serving 层会拒绝请求。verifiers 捕获此错误后设置 `state["prompt_too_long"] = True` 和 `state["is_truncated"] = True`，提前终止该 rollout。

### 5. 当前 BlenderGym 配置中**缺失**的 context 管理

| 设置 | 当前值 | 建议 |
|------|--------|------|
| `max_total_completion_tokens` | -1（禁用） | 设为 `seq_len - estimated_prompt_tokens` 以防止总 token 溢出 |
| `max_completion_tokens` | 8192（per-turn） | 可能过大，考虑降到 4096-6144 |
| Prompt 优化 | 允许长 reasoning | 考虑精简 system prompt 或禁止 reasoning |

## 当前配置状态

`configs/multimodal/rl_blendergym_kaola.toml`：
```toml
seq_len = 16384
max_completion_tokens = 8192
gpu_memory_utilization = 0.80
max_model_len = 32768
```

尚未重新提交训练。S3 `experiments/blendergym-9b-dp6` 已清空。

## 相关代码文件

| 文件 | 关键位置 | 说明 |
|------|---------|------|
| `src/prime_rl/trainer/batch.py` | L6-79 | `prepare_sample()` — 截断逻辑（BUG 所在） |
| `src/prime_rl/trainer/batch.py` | L87-144 | `packed_samples_into_micro_bs()` — 多模态样本不 pack |
| `src/prime_rl/transport/types.py` | L5-28 | `TrainingSample` 数据结构 |
| `src/prime_rl/transport/types.py` | L39-58 | `MicroBatch` 数据结构 |
| `src/prime_rl/trainer/model.py` | L907-934 | `forward()` — 多模态 forward 入口 |
| `src/prime_rl/trainer/rl/train.py` | L354-411 | 训练循环中的多模态数据加载 |
| `src/prime_rl/trainer/rl/packer.py` | L174-178 | `MultiPacker._validate_sample()` — seq_len 校验 |
| `src/prime_rl/trainer/rl/packer.py` | L106-113, L323-330 | pack 调用 `prepare_batch` |
| `src/prime_rl/trainer/models/qwen3_5_moe/modeling_qwen3_5_moe.py` | L802-831 | VLM forward — `masked_scatter` 配对逻辑 |
| `src/prime_rl/orchestrator/trajectories.py` | L231-400 | `interleave_rollout()` — 构建 TrainingSample |
| `src/prime_rl/orchestrator/trajectories.py` | L387-398 | VLM cache 挂载 pixel_values + mm_token_type_ids |
| `src/prime_rl/orchestrator/trajectories.py` | L599-637 | `_ImageStore` — 图片存储和组装 |
| `src/prime_rl/orchestrator/trajectories.py` | L720-800 | `VLMImageCache` — 累积图片缓存 |
| `src/prime_rl/orchestrator/orchestrator.py` | L491-508 | mm_token_type_ids_mapping 构建 |
| `src/prime_rl/configs/rl.py` | L622-639 | `auto_setup_seq_len()` — 唯一的 seq_len 校验 |
| `src/prime_rl/configs/orchestrator.py` | L91-132 | `max_completion_tokens` 配置 |
| `src/prime_rl/inference/vllm/serving_chat_with_tokens.py` | L152-188 | vLLM 侧 max_model_len 校验 |
| `src/prime_rl/utils/chat_template.py` | L101-105 | processor.apply_chat_template（图片→token） |
| `environments/blendergym/blendergym/env.py` | L189-316 | 多轮对话消息构建 |
| `environments/blendergym/blendergym/prompts.py` | L12-50 | System prompt |
| `configs/multimodal/rl_blendergym_kaola.toml` | 整个文件 | 训练配置 |
