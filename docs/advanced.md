# Advanced

This page covers the specialized features layered on top of the core training stack: our custom model implementations (with EP for MoE families and CP for long-context training), multimodal training, LoRA training and the multi-run manager, and the small-scale MoE testing workflow used during architecture work.

## Table of Contents

- [Custom modeling](#custom-modeling)
  - [Custom vs HF implementations](#custom-vs-hf-implementations)
  - [Expert parallelism backends](#expert-parallelism-backends)
- [Multimodal training](#multimodal-training)
  - [Supported families](#supported-families)
  - [Enabling VLM mode](#enabling-vlm-mode)
  - [Limitations](#limitations)
  - [Multi-turn training](#multi-turn-training)
- [LoRA training](#lora-training)
- [Multi-run manager](#multi-run-manager)
  - [Run discovery](#run-discovery)
  - [Eviction](#eviction)
  - [Hooks](#hooks)
- [Testing MoE at small scale](#testing-moe-at-small-scale)

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

### Multi-turn training

VLM rollouts go through the renderer-backed TITO client (`orchestrator.use_renderer = true`, required for VLMs). Per trajectory step:

1. **Render** — the renderer tokenizes messages and emits per-image multimodal tensors (`pixel_values`, `image_grid_thw` for Qwen3-VL) as `multi_modal_data`.
2. **Pack** — `interleave_rollout` concatenates per-image tensors across a sample's merged step range into a single `mm_kwargs` dict on the `TrainingSample`. Per-token `mm_token_type_ids` (0=text, 1=image, 2=video) come from `renderer.mm_token_type_id_map`.
3. **Forward** — the trainer `**`-unpacks `mm_kwargs` into the model's `forward`. Any VLM whose HF processor and forward signature agree on kwarg names works without modifying the transport.

Each multimodal sample becomes its own micro-batch (no packing) because image tensor sizes vary.

`VLLM_WORKER_MULTIPROC_METHOD=spawn` is required for VLM inference — set automatically by `uv run rl`, but if you launch `uv run inference` separately for a VLM, export it yourself.

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

LoRA pairs naturally with the multi-run manager — each run gets its own adapter, and many runs share the same backbone in trainer memory.

## Multi-run manager

`MultiRunManager` is a trainer-side singleton that lets one trainer process serve multiple concurrent orchestrator deployments, each with its own LoRA adapter, optimizer, scheduler, checkpoints, and progress tracking. Enable by setting `trainer.max_concurrent_runs > 1`.

Per-run layout under `<output_dir>/`:

```
run_abc123/
├── control/
│   ├── orch.toml                    # orchestrator config for this run
│   ├── config_validation_error.txt  # populated if validation failed
│   └── evicted.txt                  # populated if the run was evicted
├── checkpoints/
│   └── step_<N>/                    # orchestrator checkpoints
├── rollouts/
│   └── step_<N>/                    # rollouts
└── broadcast/
    └── step_<N>/                    # weight snapshots for inference
```

### Run discovery

Runs are added by dropping a `run_*` directory into `<output_dir>` with a valid `control/orch.toml`. The trainer scans periodically:

```python
multi_run_manager.discover_runs()       # master rank only
multi_run_manager.synchronize_state()   # all ranks
```

- `discover_runs()` (master): scans, filters evicted runs, detects new/deleted, validates configs, fires `discovered_hook` / `forgotten_hook`.
- `synchronize_state()` (all ranks): master broadcasts run state over the distributed store; all ranks run `deletion_hook` then `creation_hook` so DTensor allocations and other collective ops happen in lock-step.

Once `max_concurrent_runs` is reached, new `run_*` directories are ignored until existing runs are evicted or deleted.

### Eviction

The master can evict a run with `evict_run(idx, reason)`:

```python
multi_run_manager.evict_run(idx=0, reason="exceeded memory limits")
```

The eviction writes `<run_dir>/control/evicted.txt`. Effect:

- **Trainer side**: next `discover_runs()` treats the run as deleted, hooks fire, the index returns to the unused pool.
- **Orchestrator side**: checks for `evicted.txt` at the top of each iteration. If found, it raises a `RuntimeError` with the reason. The orchestrator also self-evicts after `MAX_EMPTY_BATCH_ATTEMPTS` (3) consecutive empty-batch failures, so a run with degenerate rewards doesn't sit consuming a slot forever.

### Hooks

Five hook types fire at well-defined points:

| Hook | Where | When |
|---|---|---|
| `discovered_hook` | master | new run detected and config validated |
| `forgotten_hook` | master | run deleted from the output dir |
| `config_validation_hook` | master | validate the orchestrator config when a new run is discovered |
| `creation_hook` | all ranks | after `synchronize_state` for a newly created run (use for optimizer/scheduler init, LoRA param reset) |
| `deletion_hook` | all ranks | after `synchronize_state` for a deleted run (use for releasing per-run resources) |

Deletion hooks always run before creation hooks. The creation/deletion hooks run on **all** ranks, so they're the right place for DTensor allocation and other collective work; `torch.dist.barrier()` is safe inside.

For the full API surface, see [`src/prime_rl/trainer/runs/`](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/src/prime_rl/trainer/runs). The primary use case today is the LoRA-per-run training topology — many lightweight RL runs (e.g. one per environment) sharing a single trainer process group.

## Testing MoE at small scale

When working on MoE architectures (GLM-4, Kimi, etc.), you can't iterate on a 100B+ model locally. The workflow below builds a ~0.5B model with the same architecture, warms it up with SFT, and runs RL — all on 1–2 GPUs. The goal is catching bugs in modeling code, state-dict conversions, and pipeline integration before scaling.

### Step 1: build and verify a mini model

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe
```

This creates a ~543M parameter GLM-4 MoE (1024 hidden, 24 layers, 8 experts) with random weights, copies the tokenizer from the original GLM-4 model, and verifies the HF↔PrimeRL roundtrip is lossless. To re-verify after a modeling-code change without re-creating the model:

```bash
uv run python scripts/mini_moe.py --arch glm4_moe --output-dir ./mini-glm-moe --verify-only
```

### Step 2: SFT warmup

Use the shipped debug MoE SFT config with reverse-text data:

```bash
uv run sft @ configs/debug/moe/sft/train.toml \
  --model.name ./mini-glm-moe \
  --data.name PrimeIntellect/Reverse-Text-SFT \
  --data.type null \
  --max_steps 200 \
  --optim.lr 1e-4 \
  --ckpt.weights
```

Loss drops from ~12 to ~2.5. The output won't be coherent, but the model now has a non-trivial distribution so KL divergence becomes meaningful in RL. A pre-built SFT'd checkpoint lives at [samsja/mini-glm-moe](https://huggingface.co/samsja/mini-glm-moe).

### Step 3: RL on reverse-text

```bash
uv run rl @ configs/ci/integration/reverse_text_moe/start.toml \
  --model.name samsja/mini-glm-moe \
  --trainer.model.impl custom \
  --inference.gpu-memory-utilization 0.7 \
  --inference.model.max-model-len 2048
```

What to look for:

- **No crashes.** Validates the full inference + orchestrator + trainer pipeline end-to-end.
- **Finite, non-zero KL.** Confirms the reference distribution is meaningful.
- **Loss reasonable.** Not NaN, not stuck.

Don't expect reward to climb meaningfully in 20 steps on a random model.

### Adding a new architecture

To add (e.g.) Kimi 2.5:

1. Add the modeling code under `src/prime_rl/trainer/models/<arch>/`.
2. Add a preset to `scripts/mini_moe.py` with the config class, small dimensions, HF + PrimeRL model classes, and tokenizer source:

```python
ARCH_PRESETS = {
    "glm4_moe": {
        "config_class": Glm4MoeConfig,
        "config_kwargs": dict(hidden_size=1024, num_hidden_layers=24, n_routed_experts=8, ...),
        "hf_model_class": HFGlm4MoeForCausalLM,
        "prime_model_class": PrimeRLGlm4MoeForCausalLM,
        "tokenizer_source": "THUDM/GLM-4-9B-0414",
    },
    # add your arch here
}
```

3. Run the three steps above with `--arch <your_arch>`.
