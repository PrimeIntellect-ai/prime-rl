# Advanced

This page covers the specialized features layered on top of the core training stack: our custom model implementations (with EP for MoE families and CP for long-context training), multimodal training, LoRA training, multi-tenant training, and the buffer-side difficulty controls. For developer-side workflows (adding new model architectures, debugging modeling code at small scale), see [Development](development.md).

## Table of Contents

- [Custom modeling](#custom-modeling)
  - [Custom vs HF implementations](#custom-vs-hf-implementations)
  - [Expert parallelism backends](#expert-parallelism-backends)
- [Multimodal training](#multimodal-training)
  - [Supported families](#supported-families)
  - [Enabling VLM mode](#enabling-vlm-mode)
  - [Limitations](#limitations)
- [LoRA training](#lora-training)
- [Multi-tenant training](#multi-tenant-training)
- [Difficulty pools and online filtering](#difficulty-pools-and-online-filtering)
  - [Difficulty pools](#difficulty-pools)
  - [Online difficulty filtering (ODF)](#online-difficulty-filtering-odf)

## Custom modeling

### Custom vs HF implementations

`prime-rl` ships custom optimized model implementations for several MoE families. With `model.impl = "auto"` (default) the trainer picks the custom path when the HF config type is registered, falling back to plain HF otherwise. To force one:

```toml
[trainer.model]
impl = "custom"        # or "hf" to force the HF path
```

| Family | HF config types | EP | CP |
|---|---|---|---|
| GLM-5 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8` | ✅ | ✅ |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B`, … | ✅ | ✅ |
| Qwen3.5 MoE | `Qwen/Qwen3.5-35B-A3B`, … | ✅ | ✅ |
| Qwen3 / Qwen3.5 VLMs | see [Multimodal training](#multimodal-training) | MoE only | ✅ |
| Laguna | `poolside/Laguna-XS.2` | ✅ | ✅ |
| MiniMax M2 | `MiniMax/MiniMax-M2` | ✅ | ✅ |
| Nemotron H | `nvidia/Nemotron-3-Nano-30B-A3B`, … | ✅ | ❌ |
| Trinity (AFMoE) | `arcee-ai/Trinity-Mini`, … | ✅ | ✅ |
| GLM-4 / GLM-4.5 / INTELLECT-3 | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | ✅ | ✅ |
| GPT-OSS (HF MoE) | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | ❌ | ✅ |

The custom path enables EP, selective activation checkpointing, FP8 training (`model.fp8 = true`, requires SM90+), and faster MoE kernels (`moe_use_grouped_mm = true`, default). Forcing `impl = "hf"` is mostly useful when debugging — it's slower and disables most MoE-specific knobs.

### Expert parallelism backends

`model.ep_comm_backend` picks the all-to-all kernel used for EP dispatch/combine:

- **`torch`** (default): TorchTitan's all-to-all collective. Works everywhere, no extra install.
- **`deepep`**: Custom kernels from DeepEP. Faster but requires DeepEP build (`bash scripts/install_deep_gemm.sh`, `bash scripts/install_ep_kernels.sh`) and tuning of `deepep_num_sms` (default 20) and `deepep_token_chunk_size` for your hardware.

DeepEP intranode dispatch derives the RDMA channel count as `deepep_num_sms / 2`. Lower SM count leaves more for compute; higher speeds up dispatch. Useful starting points: 16–24 SMs on H100, 20–40 on B200.

When you enable DeepEP, gradient clipping is auto-disabled (`optim.max_norm` set to `None`) because the kernels don't currently support it.

## Multimodal training

### Supported families

The built-in VLM registry covers:

| Family | `model_type` | Vision attr | LM attr |
|---|---|---|---|
| Qwen3-VL | `qwen3_vl` | `model.visual` | `model.language_model` |
| Qwen3-VL MoE | `qwen3_vl_moe` | `model.visual` | `model.language_model` |
| Qwen3.5 | `qwen3_5` | `model.visual` | `model.language_model` |
| Qwen3.5-MoE | `qwen3_5_moe` | `model.visual` | `model.language_model` |

For a model not in the table, look up the attribute paths on the loaded HF model with `model.named_children()` and set them under `[model.vlm]` directly.

### Enabling VLM mode

Add `[model.vlm]` and bfloat16 dtypes:

```toml
[model]
name = "Qwen/Qwen3-VL-4B-Instruct"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[model.vlm]
vision_encoder_attr = "model.visual"
language_model_attr = "model.language_model"
# freeze_vision_encoder = true  # default; set false to fine-tune the encoder
```

A bad attribute path errors immediately — no silent fallbacks. The weight-broadcast key prefix is derived as `{language_model_attr}.layers.` automatically.

To add a new model family permanently, append an entry to `VLM_REGISTRY` in `src/prime_rl/utils/vlm.py`.

### Limitations

- **Vision encoder frozen by default.** Set `freeze_vision_encoder = false` to fine-tune it; in that case it's FSDP-sharded per block. The combination `freeze_vision_encoder = false` + LoRA is rejected by a config validator — LoRA freezes everything non-adapter, so unfreezing the encoder under LoRA would be a silent no-op.
- **No multimodal-safe truncation.** Token sequences are truncated to `seq_len`, but `pixel_values` and `image_grid_thw` pass through unchanged. If a sample's tokens overflow, image tokens may get dropped while image tensors still describe the full image set. Set `seq_len` to cover your longest sample.
- **bfloat16 mandatory.** The trainer config validator refuses any other `optimization_dtype` / `reduce_dtype` for VLMs — vLLM serves VLMs in bfloat16 and a mismatch breaks the importance ratio.
- **Higher KL mismatch with multi-image inputs.** Expect noisier `mismatch_kl` than text-only; this is from minor numerical differences between the trainer's and vLLM's image processing.
- **Images aren't logged to monitors.** Sample logging captures the prompt text but not the actual images.

## LoRA training

LoRA is enabled by adding `[model.lora]`:

```toml
[model.lora]
rank = 16
alpha = 32
dropout = 0.0
```

`target_modules` defaults to a reasonable cross-family set (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `experts`, plus a few latent-projection names for Nemotron). Unknown names are silently ignored, so the defaults work across architectures. Add architecture-specific names to extend coverage (e.g. `in_proj` / `out_proj` for Mamba).

LoRA is supported across SFT and RL. For RL, `weight_broadcast.type = "nccl"` is **not** supported with LoRA — use the default filesystem transport. To save the raw adapter alongside the merged HF weights:

```toml
[ckpt.weights]
save_adapter_separately = true
```

LoRA pairs naturally with [multi-tenant training](#multi-tenant-training) — each tenant gets its own adapter and the backbone is shared across all of them in trainer memory.

## Multi-tenant training

Multi-tenant training lets a single trainer + inference deployment serve many concurrent LoRA "tenants" — each a fully isolated run with its own orchestrator, LoRA adapter, optimizer, scheduler, checkpoints, and progress tracking — sharing the same backbone weights and the same vLLM server. This is the topology behind hosted training on the [Prime Intellect platform (Lab)](https://app.primeintellect.ai). The trainer-side implementation is the `MultiRunManager` singleton, enabled by setting `trainer.max_concurrent_runs > 1`. For the full API surface, see [`src/prime_rl/trainer/runs/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/trainer/runs).

## Difficulty pools and online filtering

Two complementary mechanisms keep the trainer batch high-signal: **difficulty pools** that gradually retire problems the model has solved or never solves, and **online difficulty filtering (ODF)** that drops collapsed-advantage groups from the current batch.

### Difficulty pools

After each rollout, the average reward across a problem's group is compared to two thresholds:

- `buffer.easy_threshold` — at or above this, the problem moves into the `easy` pool and is no longer sampled.
- `buffer.hard_threshold` — at or below this, the problem moves into the `hard` pool and is no longer sampled.
- Otherwise the problem stays in `normal` and remains in the sampling rotation.

Pool assignments persist across checkpoints (`easy_examples.jsonl` / `hard_examples.jsonl` under each step's orchestrator checkpoint). When you resume — or want to broaden the curriculum mid-run — `buffer.easy_fraction` / `buffer.hard_fraction` randomly lift that fraction of pooled problems back into `normal` so they re-enter sampling.

```toml
[orchestrator.buffer]
easy_threshold = 0.95
hard_threshold = 0.05
easy_fraction = 0.0   # default; bump on resume to bring some easy problems back
hard_fraction = 0.0   # default; bump on resume to bring some hard problems back
```

Watch `pool/{env}/{easy,normal,hard}` (current pool ratios) and `evicted_examples/{env}/{easy,hard}` (per-step eviction rate).

### Online difficulty filtering (ODF)

`buffer.online_difficulty_filtering = true` is a per-rollout filter on the way *into* the buffer:

- Average reward across the group is **0.0** (every rollout failed) → drop the group, count under `filtered_rollouts/{env}/hard`.
- Average reward **1.0** (every rollout succeeded) → drop, count under `filtered_rollouts/{env}/easy`.
- Otherwise → into the buffer.

These are exactly the groups whose within-group advantage collapses to zero — DR-GRPO produces no gradient signal for them, so the trainer would burn step time on tokens it can't learn from.

```toml
[orchestrator.buffer]
online_difficulty_filtering = true
```

**Tradeoff: trainer stability vs. inference speed.** With ODF on, every rollout that reaches the trainer carries non-zero advantage — each trainer step's effective batch is predictable and the gradient signal is denser. The cost is paid on the inference side: rollouts get produced and then thrown away, so the orchestrator has to oversample to keep the trainer fed. If the orchestrator is your bottleneck (`time/wait_for_batch` high on the trainer), ODF can starve the loop. Bump `orchestrator.oversampling_factor` so inference produces enough groups per step to absorb the drops.

ODF is orthogonal to the pools: ODF reacts to the *current* group's reward distribution, the pools track the *running* per-problem average. Many configs use both — ODF for per-step density, pools for long-horizon curriculum cleanup.
