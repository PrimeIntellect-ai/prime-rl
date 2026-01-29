# Multimodal (VLM) Support

Prime-RL has experimental support for training vision-language models (VLMs) like Qwen3-VL.

## Current Limitations

- **No SFT support**: Supervised fine-tuning is not yet supported for VLM models. Only RL training is available.

- **No multi-turn images**: Images are only extracted from the first turn of a trajectory. Multi-turn environments that introduce new images in later turns are not supported yet.

- **Vision encoder is frozen**: The vision encoder is automatically frozen during training. Only the language model is trained.

## vLLM Configuration

When using vLLM for inference with VLM models, you must set these environment variables to avoid issues with multimodal models:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```
