# Advanced

This page covers the specialized features layered on top of the core training stack: our custom model implementations (with EP for MoE families and CP for long-context training), multimodal training, LoRA training, and disaggregated prefill/decode inference. For developer-side workflows (adding new model architectures, debugging modeling code at small scale), see [Development](development.md).

## Table of Contents

- [Custom Modeling](#custom-modeling)
  - [Router Precision](#router-precision)
  - [LM Head Precision](#lm-head-precision)
  - [Expert Parallelism Backends](#expert-parallelism-backends)
  - [Runtime Fusions](#runtime-fusions)
- [Multimodal Training](#multimodal-training)
  - [Supported Families](#supported-families)
  - [Enabling VLM Mode](#enabling-vlm-mode)
  - [Limitations](#limitations)
- [LoRA Training](#lora-training)
- [Disaggregated Prefill/Decode Inference](#disaggregated-prefilldecode-inference)

## Custom Modeling

The trainer only runs `prime-rl`'s own model implementations, selected from the HF config type. Besides dense Llama, Qwen3 and Qwen3.5, these cover the families below. Other architectures fail at trainer setup.

| Family | HF config types | EP | CP |
|---|---|---|---|
| GLM-5 / GLM-5.2 (`glm_moe_dsa`) | `zai-org/GLM-5`, `zai-org/GLM-5-FP8`, `zai-org/GLM-5.2`, `zai-org/GLM-5.2-FP8` | ✅ | ✅ |
| GLM-5.3 Flash, text only (`glm5_next`) | `zai-org/GLM-5.3-Flash-BF16` | ✅ | ✅ |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B`, … | ✅ | ✅ |
| Qwen3.5 MoE | `Qwen/Qwen3.5-35B-A3B`, … | ✅ | ✅ |
| Qwen3.5 VLMs | see [Multimodal training](#multimodal-training) | MoE only | ✅ |
| Qwen3.8 Flash Next VLM | `Qwen/Qwen3.8-Flash-Next` | ✅ | ✅ |
| Laguna | `poolside/Laguna-XS.2` | ✅ | ✅ |
| MiniMax M2 | `MiniMax/MiniMax-M2` | ✅ | ✅ |
| Muse Glimmer (text only) | `meta-models/Muse-Glimmer-30B` | — | ✅ |
| Nemotron H | `nvidia/Nemotron-3-Nano-30B-A3B`, … | ✅ | ❌ |
| Trinity (AFMoE) | `arcee-ai/Trinity-Mini`, … | ✅ | ✅ |
| GLM-4 / GLM-4.5 / INTELLECT-3 | `THUDM/GLM-4-9B-0414`, `zai-org/GLM-4.5`, `PrimeIntellect/INTELLECT-3`, … | ✅ | ✅ |
| GPT-OSS | `unsloth/gpt-oss-20b-BF16`, … | ✅ | ✅ |
| DeepSeek V4 | `deepseek-ai/DeepSeek-V4-Flash-0731` | ✅ | ✅ |

GPT-OSS uses FlashAttention 4 with learned attention sinks. Training requires SM90 or SM100/SM110 GPUs
and a BF16 checkpoint such as `unsloth/gpt-oss-20b-BF16`; the original MXFP4 checkpoints are not supported.

### Low-precision training

Dense linear precision and routed-expert precision are configured independently. `[trainer.model.quantization]` applies only to dense `Linear` modules:

- `type = "fp8"` — DeepGEMM FP8 blockwise linears (requires SM90 or SM100).
- `type = "mxfp8"` — torchao MXFP8 linears (requires SM100). `recipe` is `mxfp8_rceil` or `mxfp8_rceil_wgrad_with_hp`.

Both skip `lm_head`, the router and a few gate/projection modules by default (`ignore_patterns`). Add model-specific modules with `extra_ignore_patterns`, which keeps the defaults; setting `ignore_patterns` replaces them.

`[trainer.model.moe.compute]` selects routed-expert compute independently:

- `type = "bf16"` (default), with `backend = "torch"` (default) or `"sonicmoe"`.
- `type = "deepgemm_fp8"` (uses the DeepGEMM bundled with vLLM, `vllm.third_party.deep_gemm`; requires SM90 or SM100)
- `type = "mxfp8"` (requires `prime-kernels`, torchao, and SM100)
- `type = "nvfp4"` (requires a `prime-kernels` build with `nvfp4_moe`, and SM100/SM103: B200, GB200, B300). See [NVFP4 experts](#nvfp4-experts).

SonicMoE uses the upstream `sonic-moe` package (`uv sync --extra sonic-moe`) for fused BF16 expert computation. The supported model is Qwen3 MoE with the `gate_up` model fusion enabled. Backend selection requires fused gate/up weights, standard SwiGLU, and bias-free experts; incompatible expert structures raise an error during setup. It uses the same router and local, torch EP, or DeepEP dispatch as other compute backends:

```toml
[trainer.model]
name = "Qwen/Qwen3-30B-A3B"
ep = 2

[trainer.model.fusions]
enabled = ["gate_up"]

[trainer.model.moe.compute]
type = "bf16"
backend = "sonicmoe"

[trainer.model.moe.dispatch]
type = "torch"
```

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

All MoE compute backends accept `apply_to`:

- `"all"` (default) applies the backend to all expert groups.
- `"85%"` applies it to the first 85% of the MoE layers, rounded down; dense layers do not count. For a model with 48 MoE layers, this selects the first 40.
- `[0, 1, 2, 3]` selects explicit zero-based model layer indices; `[]` selects none.

Percentages must be between 0% and 100%; explicit indices must be within the model's layer count. Non-MoE blocks count toward layer indices, not percentages. Each selected layer uses the backend for all its routed experts. Other expert groups use `remainder` compute (BF16 by default, or MXFP8) and BF16 token transport while retaining the configured dispatch backend and expert parallelism. Dense linear quantization is configured separately.

For example, this selects routed experts in Qwen3's first four model layers:

```toml
[trainer.model.moe.compute]
type = "mxfp8"
apply_to = [0, 1, 2, 3]
```

Backend shape checks and token alignment apply only to the selected compute path.

In RL runs, `rl` quantizes the engines' routed experts like the trainer's (`model.moe.compute`, whatever the dispatch): without `inference.vllm.quantization`, it sets vLLM's `online` mode with `quantization_config.moe` (`fp8_per_block` for `deepgemm_fp8`, `mxfp8` for `mxfp8`), so only the experts are quantized. An engine quantization set by hand must give the experts the same scheme, or `rl` rejects it, since rollouts would be sampled with other numerics than the trainer computes. Dense layers are not inferred: their module names and kernels differ by model, so configure `inference.vllm.quantization` and its `ignore` list to match `[trainer.model.quantization]`, and check the selected modules on both sides. BF16 trainer experts leave the engines as configured. Quantizing a subset of MoE layers (`apply_to`) needs engine `targets` on the same layers.

#### NVFP4 experts

One switch runs routed experts in NVFP4, with one recipe after [Thinking Machines' end-to-end 4-bit training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/):

```toml
[trainer.model.moe.compute]
type = "nvfp4"

[trainer.model.fusions]
enabled = ["gate_up"]  # gate and up projections share one set of NVFP4 weight scales
```

- **Forward:** NVFP4 (E2M1 values, E4M3 scales per 16-element block). Both GEMM inputs take a per-token outer scale with each block's scale chosen by 4/6; weights take a per-expert outer scale, without 4/6. All quantizer arithmetic is IEEE.
- **Backward:** both gradient GEMMs in MXFP8 on the dequantized forward operands, so the gradients see the forward's rounding; every operand, the output gradient included, is quantized with stochastic rounding, seeded per training step so a step reproduces. Master weights, optimizer state, gradient reduction (FP32), checkpoints and weight updates stay BF16/FP32. Attention, shared experts, the router and the LM head stay BF16. LoRA and `compile.fullgraph` are not supported.
- **Layers:** by default the first 5/6 of the layers (`apply_to = "83.34%"`) run NVFP4 and the rest MXFP8 (`remainder = "mxfp8"`); `remainder = "bf16"` keeps the tail in BF16, `apply_to = "all"` runs every layer in NVFP4.
- **Dense layers:** BF16 by default, or MXFP8 with `[trainer.model.quantization] type = "mxfp8"`; `rl` adds shared experts to the trainer's ignore patterns.
- **Router replay:** `rl` turns on `trainer.enable_router_replay`, since quantization noise flips near-tied experts between sampler and trainer. Set it to `false` explicitly to turn it off.
- **Dispatch:** tokens travel to the experts already quantized (packed NVFP4 values, block scales and per-token scale, ~3.5x fewer bytes than BF16) and reach the expert GEMM without re-quantization; input gradients return in BF16. Covers local, torch all-to-all and DeepEP dispatch; `score_before_experts` is rejected. `moe.compute.dispatch = "bf16"` sends BF16 tokens instead: a temporary switch for the NVFP4-vs-BF16 dispatch comparison, to be removed once that is measured.
- **Engines:** `rl` sets `inference.moe_nvfp4`: vLLM's online `nvfp4_per_token` (FlashInfer TRTLLM) on the trainer's NVFP4 MoE layers, MXFP8 on the tail through `quantization_config.targets`, and FlashInfer's IEEE division and 4/6 (`FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH`, `FLASHINFER_NVFP4_4OVER6`; values you set win). vLLM quantizes the weights itself at load, without 4/6. Known gaps, which `rl` names in a warning: the down projection's input takes a static scale on TRTLLM, a per-token one on the trainer; with MXFP8 dense layers vLLM cannot also target the tail, so engines run NVFP4 on every MoE layer; and the division mode of vLLM's load-time weight quantizer is unverified.
- On Hopper (H100/H200) the trainer fails at setup with a pointer to `mxfp8` or `deepgemm_fp8`.
- **Mega MoE:** with `[trainer.model.moe.dispatch] type = "mega_moe"`, the same recipe runs fused: prime-mega-moe's NVFP4 Mega MoE kernel (tokens and weights quantized by prime-kernels' quantizers, the SwiGLU output quantized per token with 4/6 inside the kernel) and its MXFP8 Mega MoE backward on the dequantized operands, stochastically rounded. The MXFP8 tail runs Mega MoE's MXFP8 kernels. Differences from the unfused path: the backward recomputes the L1 output in MXFP8, so its SwiGLU backward and dW2 operand approximate the NVFP4 forward's rather than reusing it; the per-token down-projection scale needs the whole row, so L2 no longer overlaps L1 within an M block; and the stochastic-rounding bits differ (both unbiased and deterministic).

GLM-5.2 adds IndexShare: the DSA sparse-attention indexer runs only on a subset of layers and the remaining layers reuse the cached top-k indices. The trainer reads this schedule from the model's `indexer_types` config field and enables the index cache automatically, so no extra config is needed.

### Router precision

Router gates use BF16 operands and share the transformer block's FSDP unit. Logits are produced in FP32 directly from the gate GEMM;
score functions, normalization and routing weights remain FP32.

For models that weight expert outputs after expert computation, local and Torch EP dispatch multiply
by FP32 routing weights and sum the contributions in FP32, then cast once to the expert output dtype.
This also covers MXFP8 expert compute with Torch dispatch. With DeepEP, each rank sums its experts'
outputs for a token in FP32 and casts once, then DeepEP's combine adds these per-rank BF16 sums in FP32
(across nodes, after summing each node's ranks in FP32 and casting to BF16). MegaMoE uses its own combine
kernels and does not share this accumulation guarantee.

### LM head precision

The LM head computes FP32 logits from its BF16 operands: one GEMM writes the logits in FP32 instead of rounding them to BF16 (fused and vanilla head). Gradients stay BF16. This matches inference, whose `enable_fp32_lm_head` defaults to `true` (vLLM's FP32 head), and lowers the trainer–inference mismatch.

### Expert Parallelism Backends

`[trainer.model.moe.dispatch]` selects how routed tokens are dispatched and combined:

- **`torch`** (default): torch all-to-all with `transport = "bf16"` or, when MXFP8 expert compute is selected, `transport = "mxfp8"` on SM100.
- **`deepep`**: DeepEP custom dispatch/combine kernels. `num_sms` (default 32) is the number of SMs its kernels take from compute. DeepEP runs one communication channel per two SMs, so more SMs move tokens faster. Pre-built H100/H200 binaries use CUDA 13.0 and are installed by `uv sync --all-extras`.

```toml
[trainer.model.moe.dispatch]
type = "deepep"
num_sms = 32
```

### Runtime Fusions

`model.fusions` packs parameters that are always computed together into one tensor, turning several GEMMs into one. Both fusions are on by default:

- `gate_up` — each gated MoE expert's `gate_proj` and `up_proj` become one `[num_experts, 2 * intermediate_size, hidden_size]` weight, halving the routed-expert grouped GEMMs.
- `qkv` — an attention layer's `q_proj`, `k_proj` and `v_proj` (and their biases) become one linear layer.

```toml
[trainer.model.fusions]
enabled = ["gate_up", "qkv"]   # [] disables
```

Fusions are runtime-only. Checkpoints keep the canonical parameter names and shapes, so a run can turn a fusion on or off at any point and still load its own checkpoints, and exported weights are unaffected. Only modules that support a fusion are packed; a requested fusion that no module supports logs a warning and is skipped. Fusions are skipped when LoRA is enabled.

Muon receives the packed layout as matrix partitions and orthogonalizes each logical matrix on its own, so a packed parameter trains exactly as the parameters it replaces would — including per-projection learning-rate scaling for grouped-query attention — while keeping a single momentum tensor.

The experimental `shard_fused_on_dim1 = true` shards fused 2-D weights along dim 1 under FSDP, which makes weight loading and checkpointing zero-copy: fused weights and their optimizer state are read and written in place rather than assembled into a full copy on each rank first. It requires `hidden_size` to be divisible by the FSDP shard mesh size.

## Multimodal Training

### Supported Families

The built-in VLM registry covers:

| Family | `model_type` |
|---|---|
| Qwen3.5 | `qwen3_5` |
| Qwen3.5-MoE | `qwen3_5_moe` |
| Qwen3.8 Flash Next | `qwen4_exp` |

The trainer finds the vision encoder at `model.visual` and the language model at `model.language_model`.

Text-only checkpoints of these families set `language_model_only: true` in `config.json` and ship no `model.visual` weights. The trainer then builds no vision encoder. vLLM does not read this field, so for `rl` also set `inference.vllm.language_model_only = true`.

### Enabling VLM Mode

Add `[model.vlm]` to your RL or SFT config:

```toml
[model]
name = "Qwen/Qwen3.5-4B"

[model.vlm]
# freeze_vision_encoder = true  # default; set false to fine-tune the encoder
```

For `rl`, set `[model.vlm]` in the shared `[model]` block.

VLM training requires a registered custom PrimeRL implementation.

### Limitations

- **Vision encoder frozen by default.** The default LoRA targets do not match Qwen3.5 vision modules. Set `freeze_vision_encoder = false` to fine-tune the encoder; this is incompatible with LoRA because LoRA freezes all non-adapter parameters.
- **Precision settings are separate.** The examples serve with `inference.vllm.dtype = "bfloat16"` while retaining `trainer.model.optimization_dtype = "float32"` and `reduce_dtype = "float32"`. VLM mode does not require BF16 optimizer state or gradient reduction.
- **Monitor trainer–inference mismatch.** Image encoding, parallel reductions and model kernels can contribute numerical differences. Track `mismatch_kl/all/mean`, `masked_mismatch_kl/mean`, `unmasked_mismatch_kl/mean` and `is_masked/mean` alongside reward.
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

LoRA is supported across SFT and RL. NCCL weight broadcast is **not** supported with LoRA — the default NCCL transport automatically falls back to filesystem when LoRA is enabled. Broadcast dirs of LoRA runs contain the raw adapter (`adapter_model.safetensors` + `adapter_config.json`). With LoRA on, the inference server registers the adapter under the model name and serves the base model as `<model>-base`.

## Disaggregated Prefill/Decode Inference

For large MoE serving, splitting prefill and decode onto separate vLLM groups can substantially improve throughput. Pick the prefill:decode ratio based on workload shape:

| Workload | P:D ratio | Why |
|---|---|---|
| Agentic (SWE, Lean) | 3:1 | Long growing contexts → prefill-heavy |
| Non-agentic (math, chat) | 1:2 | Short prompts, long generations → decode-heavy |

Example config: [`examples/advanced/glm-5.3/swe.toml`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/advanced/glm-5.3/swe.toml) — full RL run on `GLM-5` with P/D disaggregation behind a `vllm-router`, FP8 inference, and NCCL weight broadcast, paired with an inference config from [`examples/advanced/glm-5.3/infer/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/advanced/glm-5.3/infer).

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
