# Multimodal (VLM) Support

Prime-RL has experimental support for training vision-language models (VLMs) like Qwen3-VL.

## Current Limitations

- **No SFT support**: Supervised fine-tuning is not yet supported for VLM models. Only RL training is available.

- **Vision encoder is frozen**: The vision encoder is automatically frozen during training. Only the language model is trained.

- **No multimodal-safe truncation**: Token sequences are truncated to `seq_len`, but `pixel_values` and `image_grid_thw` are passed through unchanged. If a multimodal sample exceeds `seq_len`, image tokens can be dropped while image tensors still describe the full set of images. Ensure `seq_len` covers your longest VLM samples or avoid overlong rollouts.

- **The images that the VLM sees are not logged**

- **Optimization dtype must be bfloat16**: VLM models must load in bfloat16 to match vLLM inference. If the trainer uses a different dtype, the vision encoder produces different `pixel_values`, causing a mismatch between inference and training. For now, set `optimization_dtype = "bfloat16"` and `reduce_dtype = "bfloat16"` in your trainer config.

- **Higher KL mismatch with multi-image inputs**: VLM training exhibits higher KL mismatch between inference and trainer logprobs compared to text-only models, especially with multiple images per sample. The existing importance ratio masking thresholds should handle reasonable mismatches.

## How Multi-Turn VLM Training Works

VLM training uses the same `interleave_rollout` path as text-only models. Multi-turn trajectory steps are merged into a single training sample wherever the extension property holds. When extension breaks, a new sample is started automatically.

Images are handled via a `VLMImageCache` built once per batch:

1. **Extract**: Base64 images are decoded from trajectory step prompts into PIL images
2. **Preprocess**: All images are processed in a single batched call through the HuggingFace image processor
3. **Attach**: Each training sample receives the cumulative `pixel_values` up to its last merged step

Each multimodal sample becomes its own micro-batch during training (no packing with other samples) since image tensor sizes vary per sample.

## vLLM Configuration

`VLLM_WORKER_MULTIPROC_METHOD=spawn` is required for VLM inference. This is set automatically in `src/prime_rl/inference/config.py`, so if you use `uv run rl @ ...` it works out of the box, but if you start the vLLM server yourself, make sure this environment variable is set.
