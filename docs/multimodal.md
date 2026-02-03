# Multimodal (VLM) Support

Prime-RL has experimental support for training vision-language models (VLMs) like Qwen3-VL.

## Current Limitations

- **No SFT support**: Supervised fine-tuning is not yet supported for VLM models. Only RL training is available.

- **Vision encoder is frozen**: The vision encoder is automatically frozen during training. Only the language model is trained.

- **No multimodal-safe truncation**: Token sequences are truncated to `seq_len`, but `pixel_values` and `image_grid_thw` are passed through unchanged. If a multimodal sample exceeds `seq_len`, image tokens can be dropped while image tensors still describe the full set of images. Ensure `seq_len` covers your longest VLM samples or avoid overlong rollouts.

- **The images that the VLM sees are not logged**

- **Optimization dtype must be bfloat16**: VLM models must load in bfloat16 to match vLLM inference. If the trainer uses a different dtype, the vision encoder produces different `pixel_values`, causing a mismatch between inference and training. A workaround would be to propagate the `pixel_values` computed by vLLM to the trainer, but this is more involved. For now, set `optimization_dtype = "bfloat16"` and `reduce_dtype = "bfloat16"` in your trainer config.

- **Higher KL mismatch with multi-image inputs**: VLM training exhibits higher KL mismatch between inference and trainer logprobs compared to text-only models, especially with multiple images per sample. We are investigating the root cause. The existing importance ratio masking thresholds should handle reasonable mismatches.

- **VLM requires branching strategy**: Multimodal training must use `trajectory_strategy = "branching"` in your orchestrator config. The interleaved strategy doesn't work because vLLM tokenizes images differently at different conversation states, causing token prefix mismatches between trajectory steps. Branching treats each step independently, avoiding this issue.

## vLLM Configuration

When using vLLM for inference with VLM models, you must set these environment variables to avoid issues with multimodal models:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Why Interleaved Strategy Doesn't Work for VLMs

The interleaved trajectory strategy fails for VLM conversations due to how vLLM's `/tokenize` endpoint handles images.

### The Problem

For multi-turn conversations, the interleaved strategy uses the Token-In Token-Out API:

1. **Turn 1**: Uses standard `/chat/completions` endpoint
   - vLLM properly processes images and expands `<|image_pad|>` placeholders
   - Example: 2 images → 128 tokens (64 per image based on grid dimensions)

2. **Turn 2+**: Uses `/chat/completions/tokens` with pre-tokenized prompt
   - Tokens are computed by calling vLLM's `/tokenize` endpoint
   - **`/tokenize` does NOT expand image tokens** - it returns raw `<|image_pad|>` tokens (1 per image)
   - Example: 2 images in turn 2 → only 2 tokens instead of 128

3. **Mismatch**: The model validates that `count(image_tokens) == image_features` and fails:
   ```
   ValueError: Image features and image tokens do not match: tokens: 130, features: 256
   ```

### Technical Details

- vLLM's `/tokenize` uses `apply_chat_template(tokenize=False)` followed by basic tokenization
- Image token expansion happens later in the model executor layer via `PromptReplacement`
- The HuggingFace processor can properly expand image tokens, but the tokenize endpoint doesn't use it

### Why Branching Works

The branching strategy creates separate `TrainingSample` objects per turn. Each sample goes through the standard `/chat/completions` endpoint which properly expands image tokens. Tokens and `pixel_values` are always matched per-turn.

### Configuration

```toml
[orchestrator]
trajectory_strategy = "branching"  # Required for VLM
```
