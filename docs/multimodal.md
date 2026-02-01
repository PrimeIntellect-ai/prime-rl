# Multimodal (VLM) Support

Prime-RL has experimental support for training vision-language models (VLMs) like Qwen3-VL.

## Current Limitations

- **No SFT support**: Supervised fine-tuning is not yet supported for VLM models. Only RL training is available.

- **No multi-turn images**: Images are only extracted from the first turn of a trajectory. Multi-turn environments that introduce new images in later turns are not supported yet.

- **Vision encoder is frozen**: The vision encoder is automatically frozen during training. Only the language model is trained.

- **No multimodal-safe truncation**: Token sequences are truncated to `seq_len`, but `pixel_values` and `image_grid_thw` are passed through unchanged. If a multimodal sample exceeds `seq_len`, image tokens can be dropped while image tensors still describe the full set of images. Ensure `seq_len` covers your longest VLM samples or avoid overlong rollouts.

- **The images that the VLM sees are not logged**

- **Optimization dtype must be bfloat16**: VLM models must load in bfloat16 to match vLLM inference. If the trainer uses a different dtype, the vision encoder produces different `pixel_values`, causing a mismatch between inference and training. A workaround would be to propagate the `pixel_values` computed by vLLM to the trainer, but this is more involved. For now, set `optimization_dtype = "bfloat16"` and `reduce_dtype = "bfloat16"` in your trainer config.

## vLLM Configuration

When using vLLM for inference with VLM models, you must set these environment variables to avoid issues with multimodal models:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```
