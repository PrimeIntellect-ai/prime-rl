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

### Phase 2: Inference Path (vLLM)
**Goal**: Verify vLLM can do multimodal inference with our setup

- [ ] Test vLLM with Qwen3-VL and image inputs manually
- [ ] Verify OpenAI-compatible API accepts base64 images
- [ ] Confirm logprobs are returned correctly for all tokens
- [ ] Document any config changes needed

### Phase 3: Transport Types
**Goal**: Add multimodal fields to data structures

**Files to modify:**

| File | Change |
|------|--------|
| `src/prime_rl/transport/types.py` | Add image fields to `TrainingSample` |

```python
class TrainingSample(msgspec.Struct):
    # ... existing fields ...

    # Multimodal fields (Qwen3-VL)
    pixel_values: list[list[float]] | None = None  # [num_patches, 1176] flattened
    image_grid_thw: list[list[int]] | None = None  # [num_images, 3] - T, H, W per image
```

- [ ] Add `pixel_values` and `image_grid_thw` to `TrainingSample`
- [ ] Update any serialization/deserialization as needed

### Phase 4: Orchestrator Integration
**Goal**: Orchestrator extracts and preprocesses images

**Files to modify:**

| File | Change |
|------|--------|
| `src/prime_rl/orchestrator/trajectories.py` | Extract images, call HF processor |
| `src/prime_rl/orchestrator/env_worker.py` | Pass image data through |

**Key logic:**
```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

def process_multimodal_sample(messages: list[dict], images: list[Image]) -> dict:
    # Processor handles both text and images together
    inputs = processor(
        text=processor.apply_chat_template(messages, add_generation_prompt=True),
        images=images,
        return_tensors="pt"
    )
    return {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
    }
```

- [ ] Add image extraction from OAI-spec messages
- [ ] Use HF `AutoProcessor` for preprocessing
- [ ] Store `pixel_values` and `image_grid_thw` in `TrainingSample`

### Phase 5: Trainer - Model Loading
**Goal**: Load full Qwen3-VL model with frozen vision encoder

**Files to modify:**

| File | Change |
|------|--------|
| `src/prime_rl/trainer/model.py` | Load VLM, freeze vision encoder |

```python
from transformers import Qwen3VLForConditionalGeneration

def get_model(config):
    if is_vision_model(config.model.name):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model.name,
            torch_dtype=torch.bfloat16,
        )
        # Freeze vision encoder
        for param in model.model.visual.parameters():
            param.requires_grad = False
        return model
    else:
        return AutoModelForCausalLM.from_pretrained(...)
```

**LoRA config (language model only):**
```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # LM attention layers
)
```

- [ ] Add VLM model loading path
- [ ] Freeze vision encoder parameters
- [ ] Configure LoRA for language model layers only
- [ ] Handle FSDP wrapping (vision encoder as frozen unit)

### Phase 6: Trainer - Forward Pass
**Goal**: Forward with pixel_values, let model handle DeepStack

**Files to modify:**

| File | Change |
|------|--------|
| `src/prime_rl/trainer/rl/train.py` | Pass multimodal inputs to model |
| `src/prime_rl/trainer/rl/data.py` | Handle variable-size batches (no packing) |

**Forward pass:**
```python
# Model handles everything internally:
# 1. Vision encoder: pixel_values → image_embeds + deepstack_features
# 2. Scatter image_embeds into inputs_embeds
# 3. Forward through LM with DeepStack injection
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    pixel_values=batch["pixel_values"],
    image_grid_thw=batch["image_grid_thw"],
)
logits = outputs.logits
```

**Batching without packing:**
```python
# For now, process samples individually or with simple padding
# Variable pixel_values shapes prevent efficient packing
for sample in samples:
    outputs = model(
        input_ids=sample.input_ids.unsqueeze(0),
        pixel_values=sample.pixel_values,
        image_grid_thw=sample.image_grid_thw,
    )
    # ... compute loss ...
```

- [ ] Update forward pass to include `pixel_values` and `image_grid_thw`
- [ ] Handle variable batch sizes (no packing initially)
- [ ] Verify gradients flow through LoRA layers only

### Phase 7: End-to-End Integration
**Goal**: Full RL training loop works with multimodal

**Files to create/modify:**

| File | Change |
|------|--------|
| `configs/multimodal/rl.toml` | Example config for Qwen3-VL |
| `tests/integration/test_multimodal.py` | Integration test |

- [ ] Create example config at `configs/multimodal/rl.toml`
- [ ] Test with [multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset
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
| Variable `pixel_values` shapes | Skip packing initially, process samples individually |
| DeepStack requires full model | Keep vision encoder in trainer (frozen) |
| Memory usage with vision encoder | Vision encoder is ~300M params, acceptable overhead |
| FSDP with mixed frozen/trainable | Wrap vision encoder as single frozen unit |
| Image tokens in loss | Mask `<\|image_pad\|>` tokens in loss computation |

---

## Future Improvements

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

- Support for VLMs other than Qwen3-VL
- Video input support (Qwen3-VL supports it, but we skip for now)
- Sequence packing for multimodal samples
- Vision encoder fine-tuning

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
