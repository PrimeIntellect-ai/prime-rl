# Qwen3-VL Multimodal Support Specification

## Goal

Add multimodal training support to prime-rl, specifically targeting Qwen3-VL. This is an initial implementation that intentionally overfits to Qwen3-VL - generalization to other VLMs can come later.

## TL;DR Approach

1. **Full model in trainer** - Load `Qwen3VLForConditionalGeneration` with vision encoder
2. **Freeze vision encoder** - Only train language model with LoRA
3. **Pass `pixel_values` directly** - Let model handle vision encoding + DeepStack internally
4. **No packing initially** - Variable `pixel_values` shapes don't pack well (can add later)
5. **Follow TRL's pattern** - They already support Qwen3-VL GRPO, we adapt their approach

```
vLLM: images → completion + logprobs (inference only)
                    ↓
Orchestrator: messages + images → TrainingSample
                                       ↓
Trainer: pixel_values + input_ids → [Full Model + LoRA] → logits → loss
                                          ↑
                            (vision encoder frozen, LM trained)
```

## Why This Approach (Not Pre-computed Embeddings)

### Qwen3-VL Has DeepStack

Qwen3-VL uses **DeepStack** - a technique where intermediate vision encoder features are injected into early decoder layers:

```python
# Vision encoder returns BOTH final embeddings AND intermediate features
image_embeds, deepstack_features = model.visual(pixel_values, grid_thw=...)
# deepstack_features = [layer8_feat, layer16_feat, layer24_feat]

# During text decoding, these are injected at layers 0, 1, 2
for layer_idx, decoder_layer in enumerate(self.layers):
    hidden_states = decoder_layer(hidden_states, ...)
    if layer_idx in [0, 1, 2]:
        hidden_states[visual_positions] += deepstack_features[layer_idx]
```

Pre-computing only final `image_embeds` would **lose DeepStack**, degrading quality.

### TRL Already Does This

TRL's `GRPOTrainer` supports Qwen3-VL by passing `pixel_values` directly to the model:
- Model handles vision encoding internally
- DeepStack works automatically
- No custom embedding logic needed

We follow their proven pattern.

### Packing Can Come Later

| Approach | Packing | DeepStack | Complexity |
|----------|---------|-----------|------------|
| Pre-computed embeds | ✅ Works | ❌ Lost | Medium |
| Full model (TRL-style) | ❌ Skip for now | ✅ Works | Low |
| Full DeepStack pre-compute | ✅ Works | ✅ Works | High |

We start with TRL-style (no packing), can add packing later if needed.

---

## Background

### Context
- prime-rl currently only supports text-only LLM training
- Qwen3-VL is the immediate target model for multimodal RL training
- Images in OpenAI API spec are base64-encoded in message content
- vLLM has native multimodal support
- TRL already supports Qwen3-VL GRPO (see `trl/examples/notebooks/grpo_qwen3_vl.ipynb`)

### Key Files to Modify

| Component | File | Change |
|-----------|------|--------|
| **Transport** | `src/prime_rl/transport/types.py` | Add `pixel_values`, `image_grid_thw` to `TrainingSample` |
| **Orchestrator** | `src/prime_rl/orchestrator/trajectories.py` | Extract images, preprocess with HF processor |
| **Trainer** | `src/prime_rl/trainer/model.py` | Load `Qwen3VLForConditionalGeneration`, freeze vision encoder |
| **Trainer** | `src/prime_rl/trainer/rl/train.py` | Forward with `pixel_values` + `image_grid_thw` |
| **Trainer** | `src/prime_rl/trainer/rl/data.py` | Handle variable-size `pixel_values` (no packing) |

### Qwen3-VL Model Structure

```python
Qwen3VLForConditionalGeneration
├── model: Qwen3VLModel
│   ├── visual: Qwen3VLVisionModel      # Vision encoder (frozen)
│   │   ├── patch_embed                  # 3D Conv for patches
│   │   ├── blocks[0..26]                # Vision transformer layers
│   │   ├── merger                       # Final patch merger
│   │   └── deepstack_merger_list        # Mergers for intermediate features
│   └── language_model: Qwen3VLTextModel # Language model (LoRA here)
│       ├── embed_tokens
│       ├── layers[0..31]                # Decoder layers (DeepStack injected at 0,1,2)
│       └── norm
└── lm_head: nn.Linear
```

### Key Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `input_ids` | `[batch, seq]` | Token IDs with `<\|image_pad\|>` placeholders |
| `pixel_values` | `[total_patches, 1176]` | Flattened image patches (variable per batch) |
| `image_grid_thw` | `[num_images, 3]` | Grid dims: [Temporal, Height, Width] per image |

**`image_grid_thw` explained:**
- `t` = temporal frames (1 for images, >1 for videos)
- `h` = height in patches after resize
- `w` = width in patches after resize
- Used for MRoPE (multi-dimensional rotary position embeddings)

---

## Implementation Plan

### Phase 1: Research & Understanding ✅
**Goal**: Complete deep dives, document findings

- [x] Deep dive into HF Qwen3-VL processor
- [x] Deep dive into vLLM multimodal handling
- [x] Deep dive into Qwen3-VL model architecture (including DeepStack)
- [x] Review TRL's Qwen3-VL GRPO implementation
- [x] Document findings in this spec

### Phase 2: Inference Path (vLLM) ✅
**Goal**: Verify vLLM can do multimodal inference with our setup

- [x] Test vLLM with Qwen3-VL and image inputs manually
- [x] Verify OpenAI-compatible API accepts base64 images
- [x] Confirm logprobs are returned correctly for all tokens
- [x] Document any config changes needed

**Note**: vLLM handles multimodal inference internally. Images are passed as base64 data URLs in the OpenAI-compatible chat format. vLLM processes `pixel_values` and `image_grid_thw` internally but does not return them in the response.

### Phase 3: Transport Types ✅
**Goal**: Add multimodal fields to data structures

**Files modified:**

| File | Change |
|------|--------|
| `src/prime_rl/transport/types.py` | Added `pixel_values` and `image_grid_thw` to `TrainingSample` and `MicroBatch` |

- [x] Add `pixel_values` and `image_grid_thw` to `TrainingSample`
- [x] Update any serialization/deserialization as needed

### Phase 4: Orchestrator Integration ✅
**Goal**: Orchestrator extracts and preprocesses images

**Files modified:**

| File | Change |
|------|--------|
| `src/prime_rl/orchestrator/trajectories.py` | Added `extract_images_from_prompt()` and `preprocess_images()` |
| `src/prime_rl/orchestrator/orchestrator.py` | Load `AutoProcessor` for VLM models, pass to rollout functions |
| `src/prime_rl/utils/vlm.py` | **New file** - VLM model whitelist and `is_vlm_model()` |

**Implementation:**
```python
# src/prime_rl/utils/vlm.py
SUPPORTED_VLM_PATTERNS = ["Qwen/Qwen3-VL*"]

def is_vlm_model(model_name: str) -> bool:
    return any(fnmatch.fnmatch(model_name.lower(), p.lower()) for p in SUPPORTED_VLM_PATTERNS)

# src/prime_rl/orchestrator/trajectories.py
def extract_images_from_prompt(prompt: list[dict]) -> list[Image.Image]:
    """Extract PIL images from OpenAI-format messages with base64 image_urls."""
    ...

def preprocess_images(images: list[Image.Image], processor) -> tuple[list | None, list | None]:
    """Use HF processor.image_processor to get pixel_values and image_grid_thw."""
    processed = processor.image_processor(images=images, return_tensors="pt")
    return processed["pixel_values"].tolist(), processed["image_grid_thw"].tolist()
```

- [x] Add image extraction from OAI-spec messages (base64 data URLs)
- [x] Use HF `AutoProcessor` for preprocessing
- [x] Store `pixel_values` and `image_grid_thw` in `TrainingSample`

### Phase 5: Trainer - Model Loading ✅
**Goal**: Load full Qwen3-VL model with frozen vision encoder

**Files modified:**

| File | Change |
|------|--------|
| `src/prime_rl/trainer/model.py` | VLM detection, `AutoModelForVision2Seq`, `freeze_vision_encoder()`, FSDP for vision encoder |
| `src/prime_rl/trainer/perf.py` | Handle nested `text_config` for VLM models |

**Implementation:**
```python
# Uses shared is_vlm_model() from utils/vlm.py
from prime_rl.utils.vlm import is_vlm_model

def get_model(config):
    if is_vlm_model(config.name):
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(...)
        freeze_vision_encoder(model)
        return model
    ...

def freeze_vision_encoder(model):
    # Handles both Qwen3-VL (model.model.visual) and Qwen2-VL (model.visual)
    ...
```

- [x] Add VLM model loading path
- [x] Freeze vision encoder parameters
- [x] Configure LoRA for language model layers only
- [x] Handle FSDP wrapping (vision encoder as frozen unit)

### Phase 6: Trainer - Forward Pass ✅
**Goal**: Forward with pixel_values, let model handle DeepStack

**Files modified:**

| File | Change |
|------|--------|
| `src/prime_rl/trainer/model.py` | `forward()` accepts `pixel_values` and `image_grid_thw` |
| `src/prime_rl/trainer/rl/train.py` | Pass multimodal inputs to model |
| `src/prime_rl/trainer/rl/data.py` | `TensorMicroBatch` handles multimodal fields |
| `src/prime_rl/trainer/batch.py` | `_is_multimodal_sample()` - no packing for VLM samples |

- [x] Update forward pass to include `pixel_values` and `image_grid_thw`
- [x] Handle variable batch sizes (no packing for multimodal samples)
- [x] Verify gradients flow through LoRA layers only

### Phase 7: End-to-End Integration 🔄
**Goal**: Full RL training loop works with multimodal

**Files created:**

| File | Status |
|------|--------|
| `configs/multimodal/rl.toml` | ✅ Created |
| `configs/multimodal/infer.toml` | ✅ Created |
| `environments/openmmreasoner.py` | ✅ Created (uses WeMath dataset) |
| `environments/mathvista.py` | ✅ Created |

- [x] Create example config at `configs/multimodal/rl.toml`
- [ ] Test with WeMath dataset (OpenMMReasoner-RL-74K)
- [ ] Evaluate on MathVista benchmark
- [ ] Verify metrics (loss, rewards, etc.)
- [ ] Profile memory usage and throughput

---

## Key Milestones & Success Criteria

### Milestone 1: vLLM Multimodal Inference Works
**Validation**:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model": "Qwen/Qwen3-VL-4B-Instruct", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}, {"type": "text", "text": "What is this?"}]}]}'
```
**Success**: Response includes valid completion and logprobs

### Milestone 2: Orchestrator Produces Multimodal Samples
**Validation**:
```python
sample = TrainingSample(...)
assert sample.pixel_values is not None
assert sample.image_grid_thw is not None
assert sample.pixel_values.shape[1] == 1176  # patch dim
```
**Success**: `TrainingSample` contains valid `pixel_values` and `image_grid_thw`

### Milestone 3: Trainer Forward Pass Works
**Validation**:
```python
outputs = model(
    input_ids=input_ids,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
)
assert outputs.logits.shape == (batch, seq, vocab)
loss = compute_loss(...)
loss.backward()
# Check LoRA gradients exist, vision encoder gradients are None
```
**Success**: Forward and backward pass complete, LoRA gradients computed

### Milestone 4: Single Training Step Completes
**Validation**:
- Load real multimodal batch from orchestrator
- Complete one optimizer step
- Loss is finite
- LoRA weights update
- Vision encoder weights unchanged

**Success**: `step 1/100, loss: X.XX` prints without crash

### Milestone 5: Full Training Run
**Validation**:
- Train on multimodal-open-r1-8k-verified dataset
- Evaluate on MathVista benchmark
- Reward metrics improve
- No memory leaks over time
- Checkpointing works

**Success**: WandB shows improving reward curve

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE (vLLM)                               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    /v1/chat/completions                           │   │
│  │  base64 img + text ──→ [Full Model] ──→ completion + logprobs     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼ completion + logprobs
┌─────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR                                    │
│                                                                          │
│  ┌────────────────┐     ┌─────────────────────────────────────────┐     │
│  │ verifiers env  │────▶│ HF Processor                            │     │
│  │ (OAI messages) │     │ - Extracts images from messages         │     │
│  └────────────────┘     │ - Tokenizes text with image placeholders│     │
│                         │ - Produces pixel_values, image_grid_thw │     │
│                         └──────────────────┬──────────────────────┘     │
│                                            │                            │
│                                            ▼                            │
│                         ┌──────────────────────────────────────────┐    │
│                         │ TrainingSample                           │    │
│                         │ + input_ids (with <|image_pad|> tokens)  │    │
│                         │ + pixel_values [num_patches, 1176]       │    │
│                         │ + image_grid_thw [num_images, 3]         │    │
│                         │ + logprobs, rewards, etc.                │    │
│                         └──────────────────┬───────────────────────┘    │
└────────────────────────────────────────────┼────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            TRAINER                                       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Qwen3VLForConditionalGeneration                │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Vision Encoder (FROZEN)                                     │  │   │
│  │  │                                                             │  │   │
│  │  │ pixel_values ──→ patch_embed ──→ transformer blocks ───┬───▶│ image_embeds
│  │  │                                                        │    │  │   │
│  │  │                     deepstack_merger @ layers 8,16,24 ─┴───▶│ deepstack_features
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                    │   │
│  │                              ▼                                    │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │ Language Model (LoRA)                                       │  │   │
│  │  │                                                             │  │   │
│  │  │ input_ids ──→ embed_tokens ──→ [scatter image_embeds] ─────▶│ inputs_embeds
│  │  │                                                             │  │   │
│  │  │ inputs_embeds ──→ decoder_layer_0 + deepstack[0] ──────────▶│   │   │
│  │  │              ──→ decoder_layer_1 + deepstack[1] ──────────▶│   │   │
│  │  │              ──→ decoder_layer_2 + deepstack[2] ──────────▶│   │   │
│  │  │              ──→ decoder_layers[3..31] ────────────────────▶│ hidden_states
│  │  │                                                             │  │   │
│  │  │ hidden_states ──→ lm_head ─────────────────────────────────▶│ logits
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│                    logits ──→ loss ──→ backward ──→ LoRA gradients       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Known Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Conv3d slow with torch 2.9 + cuDNN 9.8-9.14 | Override `nvidia-cudnn-cu12>=9.15` in pyproject.toml (see below) |
| Variable `pixel_values` shapes | Skip packing initially, process samples individually |
| DeepStack requires full model | Keep vision encoder in trainer (frozen) |
| Memory usage with vision encoder | Vision encoder is ~300M params, acceptable overhead |
| FSDP with mixed frozen/trainable | Wrap vision encoder as single frozen unit |
| Image tokens in loss | Mask `<\|image_pad\|>` tokens in loss computation |
| Duplicate image preprocessing | vLLM processes images for inference, orchestrator re-processes for training. See `CONSIDERATIONS.md` for potential optimization via vLLM returning `pixel_values` |
| vLLM multiprocessing + CUDA | Use `VLLM_WORKER_MULTIPROC_METHOD=spawn` (already set in verifiers) |

### Conv3d Performance Regression (torch 2.9 + cuDNN 9.8-9.14)

**Problem**: Qwen3-VL's vision encoder uses `nn.Conv3d` for patch embedding. PyTorch 2.9 disabled the fast cuDNN kernel for Conv3d due to bugs in cuDNN versions 9.8-9.14, falling back to a much slower `vol2col` kernel. This caused ~40x slowdown in vision encoder forward pass.

```python
# From transformers/models/qwen3_vl/modeling_qwen3_vl.py:68
self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)
```

**Root cause**:
- torch 2.9.1+cu128 bundles cuDNN 9.10.2 (in the buggy range)
- PyTorch's `use_cudnn()` check in `aten/src/ATen/native/Convolution.cpp` disables cuDNN for these versions
- Conv3d falls back to slow CPU-based `vol2col` kernel

**Fix**: cuDNN 9.15+ fixes the bug. We override torch's pinned dependency:

```toml
# pyproject.toml
[tool.uv]
override-dependencies = ["nvidia-cudnn-cu12>=9.15"]
```

**Verification**:
```python
import torch
print(torch.backends.cudnn.version())  # Should be 91801+ (9.18.x), not 91002 (9.10.2)
```

**Alternative for CUDA 13**: torch 2.10+cu130 bundles cuDNN 9.15.1, so no override needed. But vLLM currently requires torch 2.9, blocking this path.

**References**:
- [LlamaFactory Blog: Issues Related to Qwen3-VL](https://github.com/hiyouga/LLaMA-Factory)
- [PyTorch Issue #166122: Conv3d slow with torch 2.9](https://github.com/pytorch/pytorch/issues/166122)
- [PyTorch Issue #166790: Conv3d OOM regression](https://github.com/pytorch/pytorch/issues/166790)

---

## Future Improvements

### vLLM Returning Multimodal Data

Currently, vLLM processes images internally but doesn't return `pixel_values` and `image_grid_thw` in the response. The orchestrator re-extracts and re-processes images from prompts.

A potential optimization is to modify `serving_chat_with_tokens.py` to return these tensors, avoiding duplicate preprocessing. See `CONSIDERATIONS.md` for detailed trade-off analysis including:
- Response size concerns (multi-image prompts can have huge `pixel_values`)
- Serialization options (JSON vs compressed binary)
- Shared memory alternatives for same-node deployments

### Adding Packing Later

To enable packing while preserving DeepStack:

1. **Pre-compute all 4 tensors** in orchestrator:
   - `image_embeds` (final)
   - `deepstack_features[0]` (from layer 8)
   - `deepstack_features[1]` (from layer 16)
   - `deepstack_features[2]` (from layer 24)

2. **Store in TrainingSample**:
   ```python
   class TrainingSample(msgspec.Struct):
       image_embeds: list[list[float]] | None = None
       deepstack_features: list[list[list[float]]] | None = None  # [3, num_tokens, hidden]
       image_positions: list[int] | None = None
   ```

3. **Modify trainer** to inject deepstack features manually at layers 0,1,2

This is complex but enables both packing and DeepStack. Defer until needed.

### Vision Encoder Training

If we need to train the vision encoder:
1. Unfreeze vision encoder parameters
2. Add vision encoder layers to LoRA targets (or full fine-tune)
3. Adjust learning rates (vision encoder typically needs lower LR)

---

## Out of Scope (For Now)

- Support for VLMs other than Qwen3-VL (add to `SUPPORTED_VLM_PATTERNS` in `utils/vlm.py` when ready)
- Video input support (Qwen3-VL supports it, but we skip for now)
- Sequence packing for multimodal samples
- Vision encoder fine-tuning
- vLLM returning `pixel_values` in response (see `CONSIDERATIONS.md` for future exploration)

---

## References

- [Qwen3-VL HuggingFace](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)
- [TRL GRPO Qwen3-VL Notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/grpo_qwen3_vl.ipynb)
- [vLLM Multimodal Docs](https://docs.vllm.ai/en/latest/models/vlm.html)
- [DeepStack Paper](https://arxiv.org/abs/2406.04334)
- [multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) - Training dataset
- [MathVista](https://mathvista.github.io/) - Evaluation benchmark

---

## Appendix: Qwen3-VL DeepStack Details

### How DeepStack Works

From `transformers/models/qwen3_vl/modeling_qwen3_vl.py`:

**Vision encoder extracts intermediate features:**
```python
# Qwen3VLVisionModel.forward()
deepstack_feature_lists = []
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, ...)
    if layer_num in self.deepstack_visual_indexes:  # [8, 16, 24]
        deepstack_feature = self.deepstack_merger_list[...](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)

return hidden_states, deepstack_feature_lists
```

**Decoder injects features at early layers:**
```python
# Qwen3VLTextModel.forward()
for layer_idx, decoder_layer in enumerate(self.layers):
    hidden_states = decoder_layer(hidden_states, ...)

    # Inject deepstack features at layers 0, 1, 2
    if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
        hidden_states[visual_pos_masks] += deepstack_visual_embeds[layer_idx]
```

### Why DeepStack Matters

DeepStack provides richer visual information to early decoder layers, improving:
- Fine-grained visual understanding
- Spatial reasoning
- Object detection in complex scenes

Skipping DeepStack (by only using final `image_embeds`) would degrade these capabilities.

### Config Defaults

```python
# Qwen3VLVisionConfig
depth = 27                           # Vision transformer layers
hidden_size = 1152                   # Vision hidden dim
patch_size = 16                      # 16x16 pixel patches
spatial_merge_size = 2               # 2x2 patches merged
temporal_patch_size = 2              # For video
deepstack_visual_indexes = [8, 16, 24]  # Layers to extract features from
out_hidden_size = 3584               # Output dim (matches LM hidden)

# Qwen3VLTextConfig
num_hidden_layers = 32               # Decoder layers
hidden_size = 4096                   # LM hidden dim (4B model)
```
