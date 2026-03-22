# Multimodal (VLM) Support

Prime-RL supports training vision-language models (VLMs) like Qwen3-VL via both RL and SFT.

## Quick Start

### RL Training
```bash
uv run rl @ configs/multimodal/rl_color_codeword.toml
```

### SFT Training
```bash
# Generate synthetic data (or bring your own)
uv run python scripts/generate_color_codeword_sft_data.py

# Train
uv run sft @ configs/multimodal/sft_color_codeword.toml
```

## Configuration

### VLM Detection

VLMs are auto-detected from the model name (e.g. `Qwen/Qwen3-VL*`). For local checkpoints or custom models, set `vlm = true` explicitly:

```toml
[model]
name = "my-local-vlm-checkpoint"
vlm = true
```

### Custom VLM Architectures

For models not in the built-in registry, you can specify the vision encoder location and layer prefix:

```toml
[model]
name = "my-custom-vlm"
vlm = true
vlm_vision_encoder_attr = "model.my_vision_module"    # Dotted path to vision encoder
vlm_layer_prefix = "model.language_model.layers."      # Weight key prefix for text layers
```

The built-in registry covers:
- **Qwen3-VL** (`visual`)
- **LLaVA** (`vision_tower`)
- **Idefics3 / SmolVLM** (`vision_model`)

### Vision Encoder Training

By default, the vision encoder is frozen. To make it trainable:

```toml
[trainer.model]
freeze_vision_encoder = false
```

When unfrozen, the vision encoder is FSDP-sharded per-block for proper gradient flow. Note: this has no effect when using LoRA (LoRA freezes all non-adapter parameters).

### SFT Data Format

Your dataset needs `prompt` and `completion` columns with OpenAI-format messages. Images can be included as `image_url` content items:

```json
{
  "prompt": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "What is this?"}
    ]}
  ],
  "completion": [
    {"role": "assistant", "content": "A red square."}
  ]
}
```

Use `pack_function = "single"` for multimodal SFT — each sample becomes its own micro-batch since image tensor sizes vary.

## Current Limitations

- **Optimization dtype must be bfloat16**: VLM models must load in bfloat16 to match vLLM inference. Set `optimization_dtype = "bfloat16"` and `reduce_dtype = "bfloat16"`.

- **Multimodal samples exceeding `seq_len` are skipped**: Truncation would break the alignment between image tokens and pixel values. Ensure `seq_len` covers your longest VLM samples.

- **Per-role loss masking not supported for multimodal SFT**: The multimodal path uses a simple prompt/completion binary loss mask. `loss_mask_config` is only applied for text-only samples.

- **Higher KL mismatch with multi-image inputs**: VLM training exhibits higher KL mismatch compared to text-only, especially with multiple images.

- **Images are not logged**: The images the VLM sees during training are not logged to monitors.

## How Multi-Turn VLM RL Training Works

VLM training uses the same `interleave_rollout` path as text-only models. Multi-turn trajectory steps are merged into a single training sample wherever the extension property holds.

Images are handled via a `VLMImageCache` built once per batch:

1. **Extract**: Base64 images are decoded from trajectory step prompts into PIL images.
2. **Preprocess**: Images are processed through the HuggingFace image processor (runs in a background thread to avoid blocking the event loop).
3. **Attach**: Each training sample receives the cumulative `pixel_values` up to its last merged step.

Each multimodal sample becomes its own micro-batch during training (no packing with other samples) since image tensor sizes vary per sample.

## vLLM Configuration

`VLLM_WORKER_MULTIPROC_METHOD=spawn` is required for VLM inference. This is set automatically when using `uv run rl @ ...`, but if you start the vLLM server yourself, make sure this environment variable is set.
