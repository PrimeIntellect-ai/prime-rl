# FP8 Block-Wise Online Quantization Strategy

## Goal

Run inference on BF16 models using FP8 block-wise quantization to reduce memory usage (~2x) while maintaining accuracy. This also enables FP8 weight reloads during RL training.

## Why block-wise instead of per-tensor?

vLLM's built-in `Fp8OnlineLinearMethod` only supports **per-tensor** FP8 — one scale for the entire weight matrix. This loses precision because a single outlier can dominate the scale factor.

**Block-wise** FP8 (128x128 blocks, each with its own scale) gives much better accuracy and is compatible with pre-quantized FP8 model formats (e.g. DeepSeek, Qwen FP8 checkpoints).

## How it works

### Triton kernel (`worker/kernels/fp8.py`)

A Triton kernel partitions each 2D weight matrix into 128x128 blocks. For each block:
1. Compute the absolute max value
2. Derive a per-block scale: `scale = absmax / FP8_MAX`
3. Quantize: `qweight = clamp(weight / scale, FP8_MIN, FP8_MAX)`

Output: `(qweight: float8_e4m3fn, scale_inv: float32)` where `scale_inv` has shape `(ceil(M/128), ceil(N/128))`.

Requires **Hopper GPUs (SM90+)** — H100, H200, etc.

### Monkey patch (`patches.py`)

`monkey_patch_fp8_online_blockwise_quant()` patches two methods on vLLM's `Fp8OnlineLinearMethod`:

- **`create_weights`**: Sets `weight_block_size = [128, 128]` and creates a `W8A8BlockFp8LinearOp` for block-wise FP8 matmul dispatch.
- **`process_weights_after_loading`**: Uses our Triton kernel instead of vLLM's per-tensor path. Also runs vLLM's post-processing (AMD fnuz normalization, DeepGemm E8M0 conversion).

### Config (`configs/inference.py`)

New field on `ModelConfig`:
```toml
[model]
name = "Qwen/Qwen3-8B"
quantization = "fp8"
```

This maps to vLLM's `--quantization fp8` flag, which activates `Fp8OnlineLinearMethod` — then our monkey patch upgrades it to block-wise.

## Verification plan

Test on a Hopper machine (H100/H200):

```bash
# 1. BF16 baseline
uv run inference --model.name Qwen/Qwen3-0.6B --model.max_model_len 2048 --model.enforce-eager

# In another terminal:
uv run vf-eval reverse-text -m Qwen/Qwen3-0.6B -b http://localhost:8000/v1 -n 5 --max-tokens 256

# 2. FP8 block-wise
uv run inference --model.name Qwen/Qwen3-0.6B --model.max_model_len 2048 --model.enforce-eager --model.quantization fp8

# In another terminal:
uv run vf-eval reverse-text -m Qwen/Qwen3-0.6B -b http://localhost:8000/v1 -n 5 --max-tokens 256
```

Compare:
- Both runs should produce similar quality outputs (block-wise FP8 has minimal accuracy loss)
- FP8 run should use ~50% less GPU memory for model weights
- Check `nvidia-smi` to compare memory usage

## Future work

- **Weight reloads**: Use the same quantization path for FP8 weight updates during RL training (the kernel is already available via `quantize_weight_to_block_fp8`)
- **MoE models**: `Fp8OnlineMoEMethod` also doesn't support block quant — needs a separate patch
- **Non-Hopper GPUs**: The Triton kernel currently requires SM90+; could add a PyTorch fallback
