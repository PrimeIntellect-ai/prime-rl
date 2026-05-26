# Advanced

This page covers the specialized features layered on top of the core training stack: our custom model implementations (with EP for MoE families and CP for long-context training), multimodal training, LoRA training, multi-tenant training, and disaggregated prefill/decode inference. For developer-side workflows (adding new model architectures, debugging modeling code at small scale), see [Development](development.md).

## Table of Contents

- [Custom Modeling](#custom-modeling)
  - [Custom vs HF Implementations](#custom-vs-hf-implementations)
  - [Expert Parallelism Backends](#expert-parallelism-backends)
- [Multimodal Training](#multimodal-training)
  - [Supported Families](#supported-families)
  - [Enabling VLM Mode](#enabling-vlm-mode)
  - [Limitations](#limitations)
- [LoRA Training](#lora-training)
- [Multi-Tenant Training](#multi-tenant-training)
- [Disaggregated Prefill/Decode Inference](#disaggregated-prefilldecode-inference)

## Custom Modeling

### Custom vs HF Implementations

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

### Expert Parallelism Backends

`model.ep_comm_backend` picks the all-to-all kernel used for EP dispatch/combine:

- **`torch`** (default): TorchTitan's all-to-all collective. Works everywhere, no extra install.
- **`deepep`**: Custom kernels from DeepEP. Faster but requires DeepEP build (`bash scripts/install_deep_gemm.sh`, `bash scripts/install_ep_kernels.sh`) and tuning of `deepep_num_sms` (default 20) and `deepep_token_chunk_size` for your hardware.

DeepEP intranode dispatch derives the RDMA channel count as `deepep_num_sms / 2`. Lower SM count leaves more for compute; higher speeds up dispatch. Useful starting points: 16–24 SMs on H100, 20–40 on B200.

When you enable DeepEP, gradient clipping is auto-disabled (`optim.max_norm` set to `None`) because the kernels don't currently support it.

## Multimodal Training

### Supported Families

The built-in VLM registry covers:

| Family | `model_type` | Vision attr | LM attr |
|---|---|---|---|
| Qwen3-VL | `qwen3_vl` | `model.visual` | `model.language_model` |
| Qwen3-VL MoE | `qwen3_vl_moe` | `model.visual` | `model.language_model` |
| Qwen3.5 | `qwen3_5` | `model.visual` | `model.language_model` |
| Qwen3.5-MoE | `qwen3_5_moe` | `model.visual` | `model.language_model` |

For a model not in the table, look up the attribute paths on the loaded HF model with `model.named_children()` and set them under `[model.vlm]` directly.

### Enabling VLM Mode

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

## LoRA Training

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

## Multi-Tenant Training

Multi-tenant training lets a single trainer + inference deployment serve many concurrent LoRA "tenants" — each a fully isolated run with its own orchestrator, LoRA adapter, optimizer, scheduler, checkpoints, and progress tracking — sharing the same backbone weights and the same vLLM server. This is the topology behind hosted training on the [Prime Intellect platform (Lab)](https://app.primeintellect.ai). The trainer-side implementation is the `MultiRunManager` singleton, enabled by setting `trainer.max_concurrent_runs > 1`. For the full API surface, see [`src/prime_rl/trainer/runs/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/trainer/runs).

## Disaggregated Prefill/Decode Inference

For large MoE serving, splitting prefill and decode onto separate vLLM groups can substantially improve throughput. Pick the prefill:decode ratio based on workload shape:

| Workload | P:D ratio | Why |
|---|---|---|
| Agentic (SWE, Lean) | 3:1 | Long growing contexts → prefill-heavy |
| Non-agentic (math, chat) | 1:2 | Short prompts, long generations → decode-heavy |

Example config: [`examples/glm5_pd_disag/rl.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/glm5_pd_disag/rl.toml) — full RL run on `GLM-5` with P/D disaggregation behind a `vllm-router`, FP8 inference, and NCCL weight broadcast (see the [README](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/glm5_pd_disag) for the launch story).

Monitor live queue depths to detect imbalance:

```bash
curl -s http://<prefill_node>:8100/metrics | grep num_requests_waiting
curl -s http://<decode_node>:8200/metrics | grep num_requests_waiting
```

If prefill queues and decode is idle, add prefill nodes (and vice versa).

**UCX 1.19 requirement.** NVSHMEM needs UCX ≥ 1.19 for multi-GPU CUDA. Most clusters ship UCX 1.17 via HPC-X, which manifests as `cuStreamCreate: invalid device context` errors during DeepEP internode dispatch. Check with `/opt/hpcx/ucx/bin/ucx_info -v` and, if needed, build from source:

```bash
salloc -N 1 --gres=gpu:1 bash -c 'bash scripts/install_nixl_from_source.sh'
```

The script writes UCX 1.19 to `third_party/ucx/`; the bundled sbatch templates prepend it to `LD_LIBRARY_PATH` so it overrides the system version.
