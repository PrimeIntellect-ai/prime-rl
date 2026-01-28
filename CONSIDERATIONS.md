# Multimodal Implementation Considerations

## Current Approach

Images are extracted and preprocessed in the orchestrator (`trajectories.py`) using HuggingFace's processor after receiving trajectories from verifiers. This means:

1. vLLM processes images internally for inference (extracts `pixel_values`, `image_grid_thw`)
2. vLLM returns only text completions + logprobs
3. Orchestrator re-extracts images from base64 URLs in prompts
4. Orchestrator re-preprocesses with HF processor for training

## Future Ablation: vLLM Returning Multimodal Data

### Idea

Modify `serving_chat_with_tokens.py` to return `pixel_values` and `image_grid_thw` in the response, avoiding duplicate preprocessing.

After `_preprocess_chat()`, the `engine_prompts[0]` already contains:
```python
{
    "prompt_token_ids": [...],
    "multi_modal_data": {
        "image": {
            "pixel_values": tensor(...),
            "image_grid_thw": tensor(...),
        }
    }
}
```

### Challenges

1. **Response size** - `pixel_values` can be huge:
   - 1 image: ~256 patches × 1536 dims = 393K floats (~1.5MB)
   - 5 images: ~1.3M patches × 1536 dims = 2B floats (~8GB as JSON)

2. **Multi-image scaling** - `pixel_values` concatenates all patches:
   ```python
   # image_grid_thw tells you how to split pixel_values
   pixel_values: [512, 1536]      # 512 patches total
   image_grid_thw: [[1, 16, 16],  # image 1: 256 patches
                    [1, 16, 16]]  # image 2: 256 patches
   ```

3. **Serialization overhead** - JSON is inefficient for tensors

### Potential Solutions

1. **Compression** - Use blosc2 or similar:
   ```python
   import blosc2
   compressed = blosc2.compress(pixel_values.numpy().tobytes())
   # Often 10-50x compression for image data
   ```

2. **Shared memory** - For same-node setups, use torch multiprocessing shared tensors or file-based handoff

3. **On-demand flag** - Only return when explicitly requested:
   ```python
   class ChatCompletionRequestWithTokens(ChatCompletionRequest):
       tokens: list[int]
       return_pixel_values: bool = False
   ```

4. **Reference-based** - Return ID, store data separately:
   ```python
   {"multimodal_data_id": "abc123"}  # fetch from shared storage
   ```

### Trade-offs to Measure

| Approach | Pros | Cons |
|----------|------|------|
| Current (re-extract in orchestrator) | Simple, no vLLM changes | Duplicate preprocessing, potential inconsistency |
| vLLM returns tensors | Single preprocessing, guaranteed consistency | Large response size, serialization overhead |
| Shared memory | Zero-copy for same node | Only works same-node, complexity |
| Compressed transfer | Smaller responses | CPU overhead for compress/decompress |

### Questions to Answer

- Does duplicate preprocessing cause any inconsistency in practice?
- What's the actual overhead of re-preprocessing vs transfer?
- How many images per prompt in typical workloads?
