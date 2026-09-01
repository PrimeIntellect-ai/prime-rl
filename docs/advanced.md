# Advanced

This page covers the specialized features layered on top of the core training stack: our custom model implementations (with EP for MoE families and CP for long-context training), multimodal training, LoRA training, and disaggregated prefill/decode inference. For developer-side workflows (adding new model architectures, debugging modeling code at small scale), see [Development](development.md).

## Table of Contents

- [Custom Modeling](#custom-modeling)
  - [Expert Parallelism Backends](#expert-parallelism-backends)
- [Multimodal Training](#multimodal-training)
  - [Supported Families](#supported-families)
  - [Enabling VLM Mode](#enabling-vlm-mode)
  - [Limitations](#limitations)
- [LoRA Training](#lora-training)
- [Disaggregated Prefill/Decode Inference](#disaggregated-prefilldecode-inference)

## Custom Modeling

`prime-rl` ships custom optimized model implementations for several MoE families. With `model.impl = "auto"` (default) the trainer picks the custom path when the HF config type is registered, falling back to plain HF otherwise. To force one:

```toml
[trainer.model]
impl = "custom"        # or "hf" to force the HF path
```

| Family | HF config types | EP | CP |
|---|---|---|---|
| GLM-5 / GLM-5.2 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8`, `zai-org/GLM-5.2`, `zai-org/GLM-5.2-FP8` | ✅ | ✅ |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B`, … | ✅ | ✅ |
| Qwen3.5 MoE | `Qwen/Qwen3.5-35B-A3B`, … | ✅ | ✅ |
| Qwen3 / Qwen3.5 VLMs | see [Multimodal training](#multimodal-training) | MoE only | ❌ |
| Laguna | `poolside/Laguna-XS.2` | ✅ | ✅ |
| MiniMax M2 | `MiniMax/MiniMax-M2` | ✅ | ✅ |
| Nemotron H | `nvidia/Nemotron-3-Nano-30B-A3B`, … | ✅ | ❌ |
| Trinity (AFMoE) | `arcee-ai/Trinity-Mini`, … | ✅ | ✅ |
| GLM-4 / GLM-4.5 / INTELLECT-3 | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | ✅ | ✅ |
| GPT-OSS (HF MoE) | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | ❌ | ✅ |

Selective activation checkpointing works with either implementation. The custom path additionally enables EP, CP, low-precision training, and grouped MoE kernels. Forcing `impl = "hf"` is mostly useful when debugging and disables those model-specific runtime features.

### Low-precision training

Dense linear precision and routed-expert precision are configured independently. `[trainer.model.quantization]` applies only to dense `Linear` modules:

- `type = "fp8"` — DeepGEMM FP8 blockwise linears (requires SM90+).
- `type = "mxfp8"` — torchao MXFP8 linears (requires SM100). `recipe` is `mxfp8_rceil` or `mxfp8_rceil_wgrad_with_hp`.

`[trainer.model.moe.compute]` selects routed-expert grouped GEMMs independently:

- `type = "bf16"` (default)
- `type = "deepgemm_fp8"` (requires DeepGEMM and SM90+)
- `type = "mxfp8"` (requires `prime-kernels`, torchao, and SM100)

```toml
[trainer.model.quantization]
type = "mxfp8"
recipe = "mxfp8_rceil"

[trainer.model.moe.compute]
type = "mxfp8"
recipe = "mxfp8_rceil"

[trainer.model.moe.dispatch]
type = "torch"
transport = "mxfp8"
```

GLM-5.2 adds IndexShare: the DSA sparse-attention indexer runs only on a subset of layers and the remaining layers reuse the cached top-k indices. The trainer reads this schedule from the model's `indexer_types` config field and enables the index cache automatically, so no extra config is needed. To override the schedule manually, set `[trainer.model.index_cache]` (`topk_freq` or `topk_pattern`).

### Expert Parallelism Backends

`[trainer.model.moe.dispatch]` selects how routed tokens are dispatched and combined:

- **`torch`** (default): torch all-to-all with `transport = "bf16"` or, when MXFP8 expert compute is selected, `transport = "mxfp8"` on SM100.
- **`deepep`**: DeepEP custom dispatch/combine kernels. Set `num_sms` and optional `token_chunk_size` in the same table. Pre-built H100/H200 binaries use CUDA 13.0 and are installed by `uv sync --all-extras`.

```toml
[trainer.model.moe.dispatch]
type = "deepep"
num_sms = 20
token_chunk_size = 4096
```

With DeepEP, gradient clipping is currently not supported. (`optim.max_norm` is set to `None` automatically.)

## Multimodal Training

### Supported Families

The built-in VLM registry covers:

| Family | `model_type` | Vision attr | LM attr |
|---|---|---|---|
| Qwen3.5 | `qwen3_5` | `model.visual` | `model.language_model` |
| Qwen3.5-MoE | `qwen3_5_moe` | `model.visual` | `model.language_model` |

### Enabling VLM Mode

Add `[model.vlm]` and bfloat16 dtypes:

```toml
[model]
name = "Qwen/Qwen3.5-4B"
impl = "custom"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[model.vlm]
vision_encoder_attr = "model.visual"
language_model_attr = "model.language_model"
# freeze_vision_encoder = true  # default; set false to fine-tune the encoder
```

The weight-broadcast key prefix is derived as `{language_model_attr}.layers.` automatically.

VLM training requires a registered custom PrimeRL implementation.

### Limitations

- **Vision encoder frozen by default.** The default LoRA targets do not match Qwen3.5 vision modules. Set `freeze_vision_encoder = false` to fine-tune the encoder; this is incompatible with LoRA because LoRA freezes all non-adapter parameters.
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

LoRA is supported across SFT and RL. NCCL weight broadcast is **not** supported with LoRA — the default NCCL transport automatically falls back to filesystem when LoRA is enabled. Broadcast dirs of LoRA runs contain the raw adapter (`adapter_model.safetensors` + `adapter_config.json`).

## Disaggregated Prefill/Decode Inference

For large MoE serving, splitting prefill and decode onto separate vLLM groups can substantially improve throughput. Pick the prefill:decode ratio based on workload shape:

| Workload | P:D ratio | Why |
|---|---|---|
| Agentic (SWE, Lean) | 3:1 | Long growing contexts → prefill-heavy |
| Non-agentic (math, chat) | 1:2 | Short prompts, long generations → decode-heavy |

Example config: [`examples/advanced/glm-5.2/swe.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/advanced/glm-5.2/swe.toml) — full RL run on `GLM-5` with P/D disaggregation behind a `vllm-router`, FP8 inference, and NCCL weight broadcast, paired with an inference config from [`examples/advanced/glm-5.2/infer/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.2/infer).

Monitor live queue depths to detect imbalance:

```bash
curl -s http://<prefill_node>:8100/metrics | grep num_requests_waiting
curl -s http://<decode_node>:8200/metrics | grep num_requests_waiting
```

If prefill queues and decode is idle, add prefill nodes (and vice versa).

**Required setup for disaggregated P/D (NIXL/UCX).** The pip-wheel NIXL's bundled UCX segfaults on the prefill→decode KV transfer (`signal 11: invalid permissions for mapped object` in `libucs.so`) — reproduced on vLLM 0.22 and 0.23, with/without mooncake, with/without llm-d. Building NIXL against UCX 1.19.x from source is therefore **required** (not optional) for disaggregated P/D.

```bash
salloc -N 1 --gres=gpu:1 bash -c 'bash scripts/install_nixl_from_source.sh'
uv pip install --reinstall --no-deps deps/nixl_cu13-*.whl
```

The script writes UCX 1.19 to `third_party/ucx/`; the bundled sbatch templates prepend it to `LD_LIBRARY_PATH` so it overrides the system version. Re-run both commands after every `uv sync`, since the lock pins the wheel.
