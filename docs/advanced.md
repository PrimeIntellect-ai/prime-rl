# Advanced

This page covers the specialized features layered on top of the core training stack: our custom model implementations (with EP for MoE families and CP for long-context training), multimodal training, LoRA training, multi-tenant training, and disaggregated prefill/decode inference. For developer-side workflows (adding new model architectures, debugging modeling code at small scale), see [Development](development.md).

## Table of Contents

- [Custom Modeling](#custom-modeling)
  - [Expert Parallelism Backends](#expert-parallelism-backends)
  - [DSA Conversion](#dsa-conversion)
- [Multimodal Training](#multimodal-training)
  - [Supported Families](#supported-families)
  - [Enabling VLM Mode](#enabling-vlm-mode)
  - [Limitations](#limitations)
- [LoRA Training](#lora-training)
- [Multi-Tenant Training](#multi-tenant-training)
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
| Qwen3 / Qwen3.5 VLMs | see [Multimodal training](#multimodal-training) | MoE only | ✅ |
| Laguna | `poolside/Laguna-XS.2` | ✅ | ✅ |
| MiniMax M2 | `MiniMax/MiniMax-M2` | ✅ | ✅ |
| Nemotron H | `nvidia/Nemotron-3-Nano-30B-A3B`, … | ✅ | ❌ |
| Trinity (AFMoE) | `arcee-ai/Trinity-Mini`, … | ✅ | ✅ |
| GLM-4 / GLM-4.5 / INTELLECT-3 | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | ✅ | ✅ |
| GPT-OSS (HF MoE) | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | ❌ | ✅ |
| Kimi K2 (`kimi_k2`) | `moonshotai/Kimi-K2-Instruct`, … | ✅ | ❌ |

The custom path enables you to set EP, CP, selective activation checkpointing, low-precision training (`[trainer.model.quantization]`), and faster MoE kernels (`moe_use_grouped_mm = true`, default). Forcing `impl = "hf"` is mostly useful when debugging — it's slower and disables most MoE-specific knobs.

`kimi_k2` reuses DeepSeek-V3's architecture directly (`layers/mla.py`, dense — not sparse — Multi-head Latent Attention), since Kimi K2 is architecturally identical, down to the HF weight-key naming. It also covers Kimi K2.7's text backbone: point `model.name` at a K2.7 checkpoint and the `language_model.`-prefixed weights load with the vision tower dropped — there is no multimodal/VLM support for it.

### Low-precision training

Set `[trainer.model.quantization]` to train dense linears and MoE expert GEMMs in low precision. Two backends are available via the `type` discriminator:

- `type = "fp8"` — DeepGEMM FP8 blockwise (requires SM90+ / Hopper). Options: `enable_grouped_gemm` (FP8 MoE expert GEMM). Both default on.
- `type = "mxfp8"` — torchao MXFP8 microscaling (requires SM100+ / Blackwell). Options: `enable_grouped_gemm`, `enable_a2a` (MXFP8 expert-parallel all-to-all), and `recipe` (`mxfp8_rceil` default or `mxfp8_rceil_wgrad_with_hp`).

```toml
[trainer.model.quantization]
type = "mxfp8"
recipe = "mxfp8_rceil"
enable_a2a = true
```

GLM-5.2 adds IndexShare: the DSA sparse-attention indexer runs only on a subset of layers and the remaining layers reuse the cached top-k indices. The trainer reads this schedule from the model's `indexer_types` config field and enables the index cache automatically, so no extra config is needed. To override the schedule manually, set `[trainer.model.index_cache]` (`topk_freq` or `topk_pattern`).

### Expert Parallelism Backends

`model.ep_comm_backend` picks the all-to-all kernel used for EP dispatch/combine:

- **`torch`** (default): TorchTitan's all-to-all collective. Works everywhere, no extra install.
- **`deepep`**: Utilizes DeepEP's custom all-to-all collectives. This provides better performance if EP dimension spans multiple nodes. We provide pre-built binaries for H100/H200 with cuda runtime 12.9 installed, you can install them by running `uv sync --all-extras`.
DeepEP requires some careful tuning to achieve optimal performance, tuning parameters are `deepep_num_sms` and `deepep_token_chunk_size`.

With DeepEP, gradient clipping is currently not supported. (`optim.max_norm` is set to `None` automatically.)

### DSA Conversion

DSA-capable families (`glm_moe_dsa`, `kimi_k2_dsa`) share one attention module (`src/prime_rl/trainer/models/layers/dsa.py`) that runs in two modes, toggled by the model's `use_sparse_attn` field (default `True`): the normal DeepSeek Sparse Attention (DSA) top-k path, or an ordinary dense-causal fallback over the same MLA projections. Dense mode exists to convert an existing dense-MLA checkpoint to DSA via continued pretraining, mirroring the two-stage recipe DeepSeek (arXiv:2512.02556) and GLM (arXiv:2602.15763) used. Any dense-MLA family gets this for free by building its DSA sibling on the same module — see `kimi_k2_dsa` alongside `kimi_k2` for the pattern.

**0. Bootstrap.** A dense checkpoint has no `indexer.*` weights at all. Point `model.name` at the dense checkpoint, force the DSA config/model class (`impl = "custom"`), and set `[model.debug] random_init_indexer = true` for exactly this one run: the loader strips the (absent) indexer keys from the strict checkpoint load instead of erroring, then randomly initializes them. Turn it back off from the very next run onward — by then the checkpoint has real indexer weights.

**1. Indexer warm-up.** On `[model]`: `use_sparse_attn = false`, `freeze_all_except_indexer = true` (freezes everything *but* the indexer — plain `freeze_sparse_indexer = false` alone does not freeze the rest of the model) and `indexer_kl_coeff = <coeff>` (also flips the HF config's `train_indexer` on). Only the indexer trains, against a KL loss to the real dense attention distribution.

**2. Sparse adaptation.** `use_sparse_attn = true`, `indexer_kl_coeff = <coeff>` (still on); `freeze_all_except_indexer = false` (the default) so everything trains jointly with the ordinary LM loss; the indexer's KL target now comes from the actual sparse-selected keys instead of the full distribution.

Both stages are continued-pretraining over raw text, not chat data — use `[data] type = "raw_text"` (`RawTextDataConfig`) instead of `type = "sft"`. Run each stage as a separate `uv run sft` invocation, chained via checkpoint resume (`ckpt.resume_step`/`ckpt.resume_path`).

Once converted, recover any quality lost to sparsification with the existing `opd` algorithm: point `algorithm.teacher` at the *original* dense checkpoint (served independently — never wired into the DSA student's weight-broadcast group, since it lacks the `indexer.*` parameters) and train the converted checkpoint as the student.

**Example: converting Kimi K2 to DSA.**

First prepare a local checkpoint directory: copy `moonshotai/Kimi-K2-Instruct`'s safetensors files as-is, but replace its `config.json` with one whose `model_type` is `"kimi_k2_dsa"` (not `"kimi_k2"`) and which adds the indexer fields (`index_topk`, `index_n_heads`, `index_head_dim`, ...) — this is what tells prime-rl to load the checkpoint's (otherwise-identical) weights into the DSA-capable model class instead of the plain dense one.

```toml
# bootstrap.toml — one-off, produces the first real (randomly-initialized-indexer) checkpoint
[model]
name = "/path/to/kimi-k2-dsa-prepared"   # the copy with the edited config.json above
impl = "custom"

[model.debug]
random_init_indexer = true

[data]
type = "fake"   # any single step works; this run exists only to materialize+save the bootstrap checkpoint
```

```toml
# stage_a.toml — indexer warm-up
[model]
name = "<bootstrap.toml's output checkpoint>"
impl = "custom"
use_sparse_attn = false
freeze_all_except_indexer = true
indexer_kl_coeff = 1.0

[data]
type = "raw_text"
name = "<a long-context text corpus>"
```

```toml
# stage_b.toml — sparse adaptation, resumes from stage_a's output
[model]
name = "<stage_a.toml's output checkpoint>"
impl = "custom"
use_sparse_attn = true
indexer_kl_coeff = 1.0

[data]
type = "raw_text"
name = "<same corpus, or your real pretraining mix>"
```

Recovery afterward is `algorithm.teacher` pointed at an independently-served `moonshotai/Kimi-K2-Instruct` (the original dense checkpoint) under the `opd` algorithm, training the converted checkpoint as the student.

Kimi K2 is a 1T-parameter model — all four runs above need real multi-node compute, not a single-box smoke test.

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

The weight-broadcast key prefix is derived as `{language_model_attr}.layers.` automatically.

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

LoRA is supported across SFT and RL. For RL, NCCL weight broadcast is **not** supported with LoRA — the default NCCL transport automatically falls back to filesystem when LoRA is enabled. To save the raw adapter alongside the merged HF weights:

```toml
[ckpt.weights]
save_adapter_separately = true
```

LoRA pairs naturally with [multi-tenant training](#multi-tenant-training) — each tenant gets its own adapter and the backbone is shared across all of them in trainer memory.

## Multi-Tenant Training

Multi-tenant training lets a single trainer + inference deployment serve many concurrent LoRA "tenants" — each a fully isolated run with its own orchestrator, LoRA adapter, optimizer, scheduler, checkpoints, and progress tracking — sharing the same backbone weights and the same vLLM server. This is the topology behind hosted training on the [Prime Intellect platform (Lab)](https://app.primeintellect.ai). The trainer-side implementation is the `MultiRunManager` singleton, enabled by setting `trainer.max_concurrent_runs > 1`. For the full API surface, see [`src/prime_rl/trainer/runs.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/runs.py).

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

**Required setup for disaggregated P/D (NIXL/UCX).** The pip-wheel NIXL's bundled UCX segfaults on the prefill→decode KV transfer (`signal 11: invalid permissions for mapped object` in `libucs.so`) — reproduced on vLLM 0.22 and 0.23, with/without mooncake, with/without llm-d. Building NIXL against UCX 1.19.x from source is therefore **required** (not optional) for disaggregated P/D.

```bash
salloc -N 1 --gres=gpu:1 bash -c 'bash scripts/install_nixl_from_source.sh'
uv pip install --reinstall --no-deps deps/nixl_cu12-*.whl
```

The script writes UCX 1.19 to `third_party/ucx/`; the bundled sbatch templates prepend it to `LD_LIBRARY_PATH` so it overrides the system version. Re-run both commands after every `uv sync`, since the lock pins the wheel.