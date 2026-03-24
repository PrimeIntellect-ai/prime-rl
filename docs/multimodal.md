# Multimodal (VLM) Support

Prime-RL supports training vision-language models (VLMs) like Qwen3-VL.

## VLM Configuration

### Supported Models

The built-in registry supports these model families out of the box:

| Model Family | model_type | Vision Encoder | Language Model |
|-------------|------------|---------------|----------------|
| Qwen3-VL | `qwen3_vl` | `model.visual` | `model.language_model` |
| Qwen3.5 | `qwen3_5` | `model.visual` | `model.language_model` |
| Qwen3.5-MoE | `qwen3_5_moe` | `model.visual` | `model.language_model` |

These are auto-detected from the model name. No extra config needed:

```toml
[model]
name = "Qwen/Qwen3-VL-4B-Instruct"
```

### Local Checkpoints

For local checkpoints or renamed models whose name doesn't match a known pattern, set `vlm = true`:

```toml
[model]
name = "/path/to/my-qwen3-vl-finetune"
vlm = true
```

### Custom VLM Architectures

For VLM families **not in the registry**, you must tell prime-rl where the vision encoder and language model live on the model object:

```toml
[model]
name = "my-org/my-custom-vlm"
vlm = true
vlm_vision_encoder_attr = "model.vision_tower"       # dotted path to vision encoder
vlm_language_model_attr = "model.language_model"      # dotted path to language model (must have .layers)
```

Both fields accept dotted attribute paths resolved on the loaded model. A bad path (e.g. a typo) raises a `ValueError` immediately — there is no silent fallback.

The weight key prefix for NCCL broadcasting is derived automatically as `{vlm_language_model_attr}.layers.`.

To add permanent support for a new model family, add an entry to `VLM_REGISTRY` in `src/prime_rl/utils/vlm.py`.

### Detection Rules

VLM detection follows this order — first match wins:

1. **Explicit `vlm = true/false`** in config — always respected, no auto-detection
2. **Model name pattern** — checked against `SUPPORTED_VLM_PATTERNS` (e.g. `Qwen/Qwen3-VL*`)
3. **Otherwise** — treated as text-only

There are no silent fallbacks. If your model is a VLM but isn't detected, set `vlm = true`.

## Current Limitations

- **Vision encoder is frozen**: The vision encoder is automatically frozen during training. Only the language model is trained.

- **No multimodal-safe truncation**: Token sequences are truncated to `seq_len`, but `pixel_values` and `image_grid_thw` are passed through unchanged. If a multimodal sample exceeds `seq_len`, image tokens can be dropped while image tensors still describe the full set of images. Ensure `seq_len` covers your longest VLM samples.

- **Optimization dtype must be bfloat16**: Set `optimization_dtype = "bfloat16"` and `reduce_dtype = "bfloat16"` in your trainer config.

- **Higher KL mismatch with multi-image inputs**: VLM training exhibits higher KL mismatch compared to text-only, especially with multiple images.

- **Images are not logged**: The images the VLM sees during training are not logged to monitors.

## How Multi-Turn VLM RL Training Works

VLM training uses the same `interleave_rollout` path as text-only models. Multi-turn trajectory steps are merged into a single training sample wherever the extension property holds.

Images are handled via a `VLMImageCache` built once per batch:

1. **Extract**: Base64 images are decoded from trajectory step prompts into PIL images.
2. **Preprocess**: Images are processed through the HuggingFace image processor, producing `pixel_values` and `image_grid_thw`.
3. **Attach**: Each training sample receives the cumulative `pixel_values` up to its last merged step.

Each multimodal sample becomes its own micro-batch during training (no packing) since image tensor sizes vary.

## vLLM Configuration

`VLLM_WORKER_MULTIPROC_METHOD=spawn` is required for VLM inference. This is set automatically when using `uv run rl @ ...`, but if you start the vLLM server yourself, make sure this environment variable is set.
