# Qwen3-VL Multimodal Support Specification

## Goal

Add multimodal training support to prime-rl, specifically targeting Qwen3-VL. This is an initial implementation that intentionally overfits to Qwen3-VL - generalization to other VLMs can come later.

## TL;DR Approach

1. **Freeze vision encoder** - only train language model with LoRA
2. **vLLM computes image embeddings** - add `/encode_images` endpoint (vision encoder already loaded)
3. **Orchestrator stores embeddings** - `TrainingSample` gets `image_embeds` field
4. **Packer works normally** - `image_embeds` have uniform shape like text embeddings
5. **Trainer uses `inputs_embeds`** - no vision encoder needed, just LM + LoRA

```
vLLM: images → /encode_images → image_embeds
                                      ↓
Orchestrator: token_ids + image_embeds → TrainingSample
                                              ↓
Packer: text_embeds + image_embeds → packed inputs_embeds
                                              ↓
Trainer: inputs_embeds → LM + LoRA → logits → loss
```

## Background

### Context
- prime-rl currently only supports text-only LLM training
- Qwen3-VL is the immediate target model for multimodal RL training
- Images in OpenAI API spec are base64-encoded in message content
- vLLM has native multimodal support
- verifiers follows OAI spec for image handling

### Key Files to Modify

| Component | File | Current Purpose |
|-----------|------|-----------------|
| **Inference** | `src/prime_rl/inference/vllm/server.py` | vLLM server with custom endpoints (`/update_weights`, etc.) |
| **Inference** | `src/prime_rl/inference/vllm/worker/filesystem.py` | Worker extension for weight updates |
| **Transport** | `src/prime_rl/transport/types.py` | `TrainingSample`, `MicroBatch` data structures |
| **Orchestrator** | `src/prime_rl/orchestrator/trajectories.py` | Converts rollouts → `TrainingSample` |
| **Orchestrator** | `src/prime_rl/orchestrator/env_worker.py` | Runs verifiers environments, extracts results |
| **Packer** | `src/prime_rl/trainer/batch.py` | Packs samples into micro-batches (FFD bin packing) |
| **Trainer** | `src/prime_rl/trainer/model.py` | Model loading, FSDP setup |
| **Trainer** | `src/prime_rl/trainer/rl/train.py` | Training loop, forward pass |
| **Trainer** | `src/prime_rl/trainer/rl/loss.py` | Loss computation with masking |
| **Trainer** | `src/prime_rl/trainer/rl/data.py` | Data loading, `TensorMicroBatch` |

### Key Technical Understanding (from Deep Dives)

**How images become tokens:**
1. Image is preprocessed into `pixel_values: [num_patches, 1176]` (variable based on resolution)
2. Vision encoder converts to `image_embeds: [num_tokens, 3584]` (num_tokens = num_patches/4)
3. Text contains `<|image_pad|>` placeholder tokens (token ID from model config)
4. Model replaces `<|image_pad|>` embeddings with `image_embeds` via `masked_scatter`

**Why packing works with embeddings:**
- `pixel_values` has variable shape (depends on image resolution) → hard to pack
- `image_embeds` has uniform hidden_dim (same as text) → packs like text

**vLLM multimodal:**
- Accepts images via OAI API: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
- Uses HF processor internally (`Qwen3VLMultiModalProcessor`)
- Model's `visual` attribute is the vision encoder: `model.visual(pixel_values, grid_thw=...)`
- Already supports `image_embeds` as input (skips vision encoder if provided)

**Model structure (Qwen3-VL):**
```
Qwen2_5_VLForConditionalGeneration
├── model.visual          # Vision encoder (ViT) - we freeze this
├── model.language_model  # Language model (Qwen2) - we train with LoRA
└── lm_head               # Vocab projection
```

---

## Technical Deep Dives Required

### Deep Dive 1: HuggingFace Processor for Qwen3-VL ✅

**Location**: `.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/`

**Questions answered**:
- [x] What class handles Qwen3-VL processing? → `Qwen2_5_VLProcessor` (wraps image_processor + tokenizer)
- [x] What does `processor(text, images)` return? → `BatchFeature` with `input_ids`, `pixel_values`, `image_grid_thw`
- [x] How are image tokens represented in `input_ids`? → `<|image_pad|>` expanded to variable count based on image size
- [x] What additional tensors are needed? → `pixel_values: [num_patches, hidden_dim]`, `image_grid_thw: [num_images, 3]`
- [x] How does the processor's tokenizer differ from `AutoTokenizer`? → Same tokenizer, processor just adds image handling
- [x] Can we use processor for training data preparation? → YES, processor handles both text + images together

### Deep Dive 2: vLLM Multimodal Support ✅

**Location**: `.venv/lib/python3.12/site-packages/vllm/`

**Questions answered**:
- [x] How does vLLM's OpenAI-compatible API accept images? → Standard OAI format: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`
- [x] Does vLLM use HF's processor internally? → YES, `Qwen3VLMultiModalProcessor` calls HF's `Qwen3VLProcessor`
- [x] What format does vLLM expect for `pixel_values`? → Same as HF: `[num_patches, hidden_dim]`
- [x] Are there any vLLM-specific multimodal configs we need? → NO, works out of the box
- [x] How does vLLM return logprobs for image tokens? → Returns logprobs for ALL tokens including `<|image_pad|>` placeholders

### Deep Dive 3: Qwen3-VL Model Architecture ✅

**Questions answered**:
- [x] What is the model structure? → `model.visual` (ViT), `model.language_model` (Qwen2), `lm_head`
- [x] How does the vision encoder connect to the LLM? → Image embeddings replace `<|image_pad|>` token embeddings via `masked_scatter`
- [x] What are the forward pass signature requirements? → `input_ids`, `pixel_values`, `image_grid_thw` (position_ids auto-computed)
- [x] How should FSDP wrapping handle the vision encoder? → Wrap separately or freeze entirely

---

## Implementation Plan

### Phase 1: Research & Understanding ✅
**Goal**: Complete deep dives, document findings

- [x] Deep dive into HF Qwen3-VL processor
- [x] Deep dive into vLLM multimodal handling
- [x] Deep dive into Qwen3-VL model architecture
- [x] Document findings in this spec (see Appendix)

### Phase 2: Inference Path (vLLM)
**Goal**: Verify vLLM can do multimodal inference with our setup

- [ ] Test vLLM with Qwen3-VL and image inputs manually
- [ ] Verify OpenAI-compatible API accepts base64 images
- [ ] Confirm logprobs are returned correctly for all tokens
- [ ] Document any config changes needed

### Phase 3: vLLM Image Embedding Endpoint
**Goal**: Add endpoint to extract image embeddings from vLLM

**Files to modify:**

| File | Change |
|------|--------|
| `src/prime_rl/inference/vllm/worker/filesystem.py` | Add `encode_images()` method |
| `src/prime_rl/inference/vllm/server.py` | Add `/encode_images` endpoint |

**Worker extension** (`src/prime_rl/inference/vllm/worker/filesystem.py`):
```python
def encode_images(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    """Encode images using the model's vision encoder."""
    model_runner = self.model_runner
    if hasattr(model_runner.model, "runnable"):
        model = model_runner.model.runnable
    else:
        model = model_runner.model

    # Call vision encoder directly
    with torch.no_grad():
        image_embeds = model.visual(pixel_values.to(model.visual.dtype), grid_thw=image_grid_thw)
    return image_embeds
```

**Server endpoint** (`src/prime_rl/inference/vllm/server.py`):
```python
@router.post("/encode_images")
async def encode_images(request: Request):
    data = await request.json()
    images = data["images"]  # List of base64 data URLs

    # Use HF processor to convert images → pixel_values
    processor = AutoProcessor.from_pretrained(model_name)
    inputs = processor(images=decode_images(images), return_tensors="pt")

    # Call worker to compute embeddings
    image_embeds = await engine_client(request).collective_rpc(
        "encode_images",
        args=(inputs["pixel_values"], inputs["image_grid_thw"])
    )
    return {"image_embeds": image_embeds.tolist(), "image_grid_thw": inputs["image_grid_thw"].tolist()}
```

- [ ] Add `encode_images` method to `src/prime_rl/inference/vllm/worker/filesystem.py`
- [ ] Add `/encode_images` endpoint to `src/prime_rl/inference/vllm/server.py`
- [ ] Handle image preprocessing (base64 → pixel_values) using HF processor
- [ ] Test endpoint returns correct embeddings

### Phase 4: Orchestrator Integration
**Goal**: Orchestrator calls vLLM to get image embeddings

**Files to modify:**

| File | Change |
|------|--------|
| `src/prime_rl/transport/types.py` | Add `image_embeds` field to `TrainingSample` |
| `src/prime_rl/orchestrator/trajectories.py` | Extract images, call `/encode_images`, store embeds |
| `src/prime_rl/orchestrator/env_worker.py` | Pass image data through `extract_result()` |

**Transport types** (`src/prime_rl/transport/types.py`):
```python
class TrainingSample(msgspec.Struct):
    # ... existing fields ...
    image_embeds: list[list[float]] | None = None  # [num_image_tokens, hidden_dim]
    image_positions: list[int] | None = None  # Token positions where images go
```

- [ ] Add `image_embeds` and `image_positions` to `TrainingSample` in `src/prime_rl/transport/types.py`
- [ ] Extract images from OAI-spec prompts in `src/prime_rl/orchestrator/trajectories.py`
- [ ] Call `/encode_images` endpoint on vLLM server
- [ ] Store `image_embeds` in trajectory data

### Phase 5: Packer - Pack with Image Embeddings
**Goal**: Packer handles mixed text + image embedding sequences

**Files to modify:**

| File | Change |
|------|--------|
| `src/prime_rl/trainer/batch.py` | Modify packing to handle `image_embeds` |
| `src/prime_rl/transport/types.py` | Add `inputs_embeds` to `MicroBatch` |

**Key insight**: Packing currently works on `token_ids`. For multimodal, we need to:
1. Convert `token_ids` → `text_embeds` using embedding layer
2. Insert `image_embeds` at `image_positions`
3. Pack the combined `inputs_embeds`

- [ ] Modify `pack_samples_into_micro_bs()` in `src/prime_rl/trainer/batch.py`
- [ ] Add `inputs_embeds` field to `MicroBatch` in `src/prime_rl/transport/types.py`
- [ ] Handle embedding layer lookup (need access to model's embed layer)

### Phase 6: Trainer - Forward with Embeddings
**Goal**: Trainer receives embeddings directly, applies LoRA

**Files to modify:**

| File | Change |
|------|--------|
| `src/prime_rl/trainer/model.py` | Load LM only for VLM (no vision encoder) |
| `src/prime_rl/trainer/rl/train.py` | Forward with `inputs_embeds` instead of `input_ids` |
| `src/prime_rl/trainer/rl/loss.py` | Handle image token masking in loss |

**Model loading** (`src/prime_rl/trainer/model.py`):
```python
def get_model(config):
    if is_vision_model(config.model.name):
        # Load full model but only use language_model for training
        full_model = AutoModelForVision2Seq.from_pretrained(...)
        model = full_model.language_model  # Just the LM, no vision encoder
        # Or: freeze vision encoder if we need it for embedding lookup
    else:
        model = AutoModelForCausalLM.from_pretrained(...)
```

**Forward pass** (`src/prime_rl/trainer/rl/train.py`):
```python
# Instead of: logits = model(input_ids=input_ids, ...)
# Use: logits = model(inputs_embeds=inputs_embeds, ...)
```

- [ ] Update model loading in `src/prime_rl/trainer/model.py` for VLMs
- [ ] Update forward pass in `src/prime_rl/trainer/rl/train.py` to use `inputs_embeds`
- [ ] Verify LoRA works on language model layers
- [ ] Update loss masking in `src/prime_rl/trainer/rl/loss.py` for image tokens

### Phase 7: End-to-End Integration
**Goal**: Full RL training loop works with multimodal

**Files to create/modify:**

| File | Change |
|------|--------|
| `configs/multimodal/rl.toml` | Example config for Qwen3-VL |
| `tests/integration/test_multimodal.py` | Integration test |

- [ ] Create example config at `configs/multimodal/rl.toml`
- [ ] Train on [OpenMMReasoner-RL-74K](https://huggingface.co/datasets/OpenMMReasoner/OpenMMReasoner-RL-74K) dataset
- [ ] Evaluate on MathVista benchmark
- [ ] Verify metrics (loss, rewards, etc.)
- [ ] Profile memory usage and throughput

---

## Key Milestones & Success Criteria

### Milestone 1: vLLM Multimodal Inference Works
**Validation**:
```bash
# Can send image to vLLM and get valid response + logprobs
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"model": "Qwen/Qwen2.5-VL-7B-Instruct", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```
**Success**: Response includes valid completion and logprobs

### Milestone 2: vLLM /encode_images Endpoint Works
**Validation**:
```bash
# Can call endpoint and get embeddings
curl -X POST http://localhost:8000/encode_images \
  -d '{"images": ["data:image/png;base64,..."]}'
# Returns: {"image_embeds": [[...]], "image_grid_thw": [[1, 16, 16]]}
```
```python
# Embeddings have correct shape
assert image_embeds.shape == (num_image_tokens, hidden_dim)  # e.g., (256, 3584)
```
**Success**: Endpoint returns valid embeddings

### Milestone 3: Orchestrator Gets Embeddings from vLLM
**Validation**:
```python
# Orchestrator calls vLLM for embeddings
image_embeds = await client.post("/encode_images", images=images)

# TrainingSample contains embeddings
sample = TrainingSample(...)
assert sample.image_embeds is not None
assert sample.image_embeds.shape == (num_tokens, hidden_dim)
```
**Success**: Can print `TrainingSample` with valid `image_embeds`

### Milestone 4: Packer Handles Image Embeddings
**Validation**:
```python
# Packer produces inputs_embeds (not input_ids)
micro_batch = pack_samples(samples)
assert micro_batch.inputs_embeds.shape == (seq_len, hidden_dim)
assert micro_batch.input_ids is None  # Using embeds directly
```
**Success**: Packed batch contains correct `inputs_embeds` with text + image

### Milestone 5: Trainer Forward with Embeddings
**Validation**:
```python
# Forward with inputs_embeds (no vision encoder needed)
outputs = model.language_model(
    inputs_embeds=inputs_embeds,
    position_ids=position_ids,
)
assert outputs.logits.shape == (batch, seq, vocab)
loss = compute_loss(...)  # No NaN/Inf
loss.backward()  # Gradients flow through LoRA
```
**Success**: Forward and backward pass, LoRA gradients computed

### Milestone 6: Single Training Step Completes
**Validation**:
- Load real multimodal batch from orchestrator
- Packer produces valid `inputs_embeds`
- Complete one optimizer step
- Loss is finite
- LoRA weights update

**Success**: `step 1/100, loss: X.XX` prints without crash

### Milestone 7: Full Training Run
**Validation**:
- Train on OpenMMReasoner-RL-74K dataset
- Evaluate on MathVista benchmark
- Reward metrics improve (or at least don't degrade)
- No memory leaks over time
- Checkpointing works

**Success**: WandB shows improving reward curve, MathVista scores improve

---

## Simplified Approach: Frozen Vision + Pre-computed Embeddings

**Key Decision**: Freeze vision encoder, use LoRA on language model only.

This unlocks a much simpler architecture:

### Why This Is Easier

| Aspect | With Vision Training | Frozen + Pre-computed Embeds |
|--------|---------------------|------------------------------|
| Trainer memory | Vision encoder (~600MB) + LLM | Just LLM |
| FSDP complexity | Must wrap vision encoder | Standard LLM wrapping |
| Packing | Can't pack (variable pixel_values shapes) | **CAN pack** (uniform embed shapes) |
| Data flow | pixel_values through trainer | image_embeds directly to LLM |
| LoRA | Complex (which layers?) | Simple (language model only) |

### Architecture

```
ORCHESTRATOR                          TRAINER
─────────────────────────────────     ─────────────────────────────────

images ──→ [Vision Encoder] ──→ image_embeds ──┐
              (frozen)                         │
                                               ▼
text ────→ [Tokenizer] ───────→ token_ids ───→ [Embed] ──→ [Pack] ──→ inputs_embeds
                                                              │
                                                              ▼
                                                    [Language Model + LoRA]
                                                              │
                                                              ▼
                                                          logits
```

### Key Insight: Packing Works!

```python
# image_embeds have uniform hidden_dim (same as text embeddings)
image_embeds: [num_image_tokens, 3584]  # uniform, packable!

# vs pixel_values which vary by resolution
pixel_values: [num_patches, 1176]  # variable num_patches, hard to pack
```

Since `image_embeds` are just embeddings (like text embeddings), standard packing works:
```
Sample A: [text_embeds | image_embeds]  →  [356, 3584]
Sample B: [text_embeds | image_embeds]  →  [512, 3584]
Packed:   [A | B]                       →  [868, 3584]  ✓
```

---

## Known Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Image tokens vary in count | Let images "pay" in tokens, no explicit cap |
| Different tokenization for VLM | Use processor instead of tokenizer |
| Re-tokenization in multi-turn | Start with single-turn environments |
| Need vision encoder for embeddings | vLLM already has it loaded, expose via `/encode_images` |
| Embedding layer needed for packing | Load embed layer in packer (small, ~500MB for 7B model) |

---

## Out of Scope (For Now)

- Support for VLMs other than Qwen3-VL
- Video input support
- Vision encoder fine-tuning (frozen for now, can add later)

---

## References

- [Qwen2-VL HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [vLLM Multimodal Docs](https://docs.vllm.ai/en/latest/models/vlm.html)
- [PR #1463 (PoC, for reference only)](https://github.com/PrimeIntellect-ai/prime-rl/pull/1463)
- [OpenMMReasoner-RL-74K](https://huggingface.co/datasets/OpenMMReasoner/OpenMMReasoner-RL-74K) - Training dataset (74K multimodal RL examples)
- [MathVista](https://mathvista.github.io/) - Evaluation benchmark

---

## Appendix: Deep Dive Findings

### A. HuggingFace Processor Findings

**Location**: `.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/`

#### Key Classes

1. **`Qwen2_5_VLProcessor`** (`processing_qwen2_5_vl.py`)
   - Wraps: `image_processor` + `tokenizer` + `video_processor`
   - Attributes: `image_token = "<|image_pad|>"`, `video_token = "<|video_pad|>"`

2. **`Qwen2VLImageProcessor`** (`image_processing_qwen2_vl.py`)
   - Handles image preprocessing, resizing, normalization
   - Default params: `patch_size=14`, `temporal_patch_size=2`, `merge_size=2`

#### Processor `__call__` Signature

```python
processor(
    images: ImageInput = None,       # PIL.Image, np.ndarray, or list
    text: str | list[str] = None,    # Text with <|image_pad|> placeholders
    videos: VideoInput = None,
    return_tensors: str = "pt",
    **kwargs
) -> BatchFeature
```

#### Output `BatchFeature` Contains

| Field | Shape | Description |
|-------|-------|-------------|
| `input_ids` | `[batch, seq]` | Token IDs including expanded image placeholders |
| `attention_mask` | `[batch, seq]` | Standard attention mask |
| `pixel_values` | `[num_patches, hidden_dim]` | Flattened image patches |
| `image_grid_thw` | `[num_images, 3]` | Grid dimensions (temporal, height, width) |

#### How Image Tokens Work

1. Text must contain `<|image_pad|>` placeholder (one per image)
2. Processor computes: `num_image_tokens = image_grid_thw[i].prod() // merge_size**2`
3. Each `<|image_pad|>` is expanded to `num_image_tokens` tokens
4. The number of tokens **depends on image resolution** (dynamic)

#### pixel_values Shape Calculation

```python
# For each image:
resized_h, resized_w = smart_resize(h, w, factor=patch_size*merge_size, min_pixels, max_pixels)
grid_t = 1  # for images (videos have grid_t > 1)
grid_h = resized_h // patch_size
grid_w = resized_w // patch_size
num_patches = grid_t * grid_h * grid_w
hidden_dim = channels * temporal_patch_size * patch_size * patch_size  # 3 * 2 * 14 * 14 = 1176

# Final shape: [sum(num_patches for all images), hidden_dim]
```

#### Key Insight for Training

The processor handles both tokenization AND image preprocessing together. We should:
1. Use `AutoProcessor.from_pretrained()` instead of `AutoTokenizer`
2. Pass both text and images to get correctly aligned `input_ids` and `pixel_values`
3. The processor's tokenizer IS the model's tokenizer (same vocab)

---

### B. vLLM Multimodal Findings

**Location**: `.venv/lib/python3.12/site-packages/vllm/`

#### How vLLM Receives Images (OpenAI API)

vLLM accepts images via the standard OpenAI chat completions format:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64_data>"}},
      {"type": "text", "text": "What is this?"}
    ]
  }]
}
```

#### Image Processing Flow in vLLM

1. **`chat_utils.py`**: Parses message content, extracts image URLs
2. **`multimodal/utils.py:MediaConnector._load_data_url()`**: Decodes base64 data URLs
3. **`multimodal/image.py:ImageMediaIO.load_base64()`**: Converts to PIL Image
4. **`model_executor/models/qwen3_vl.py:Qwen3VLMultiModalProcessor`**: Calls HF processor

#### vLLM Qwen3-VL Model

```python
# vllm/model_executor/models/qwen3_vl.py
class Qwen3VLMultiModalProcessor(BaseMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        processor = self.info.get_hf_processor(**mm_kwargs)  # Uses HF's Qwen3VLProcessor
        # ... processes images and text together
```

#### Key vLLM Findings

1. **vLLM uses HF's processor internally** - no custom image processing
2. **Logprobs are returned for ALL tokens** including image placeholder tokens
3. **No changes needed on inference side** - vLLM handles multimodal natively
4. The `<|image_pad|>` tokens get proper logprobs (though they're not trained)

---

### C. Qwen3-VL Architecture Findings

**Location**: `.venv/lib/python3.12/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

#### Model Structure

```python
Qwen2_5_VLForConditionalGeneration
├── model: Qwen2_5_VLModel
│   ├── visual: Qwen2_5_VisionTransformer  # Vision encoder
│   └── language_model: Qwen2Model         # Language model
└── lm_head: nn.Linear                     # Vocab projection
```

#### Forward Signature

```python
def forward(
    self,
    input_ids: torch.LongTensor,              # [batch, seq] - includes image tokens
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],  # Can be auto-computed
    pixel_values: Optional[torch.Tensor],      # [num_patches, hidden_dim]
    image_grid_thw: Optional[torch.LongTensor],# [num_images, 3]
    pixel_values_videos: Optional[...],        # For videos
    video_grid_thw: Optional[...],
    # ... other standard args
) -> Qwen2_5_VLCausalLMOutputWithPast
```

#### How Vision is Merged with Text

```python
# In Qwen2_5_VLModel.forward():

# 1. Embed text tokens
inputs_embeds = self.get_input_embeddings()(input_ids)

# 2. Compute image features from pixel_values
if pixel_values is not None:
    image_embeds = self.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_embeds, dim=0)

    # 3. Find image placeholder positions and replace with image embeddings
    image_mask = (input_ids == self.config.image_token_id)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

# 4. Forward through language model
outputs = self.language_model(inputs_embeds=inputs_embeds, ...)
```

#### FSDP Considerations

1. **Vision encoder (`model.visual`)** should be wrapped separately or frozen
2. **Language model** uses standard FSDP wrapping
3. The vision encoder is relatively small (~300M params for ViT-G)
4. Consider freezing vision encoder initially to simplify training

#### Properties for Accessing Components

```python
model.visual          # Vision encoder
model.language_model  # Language model (Qwen2)
model.lm_head        # LM head for logits
```

---

### D. Data Flow Summary (vLLM provides Image Embeddings)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE (vLLM)                               │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    /v1/chat/completions                           │  │
│  │  base64 img ──→ [Vision Encoder] ──→ [Language Model] ──→ completion │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              │ (reuse vision encoder!)                   │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    /encode_images (NEW)                           │  │
│  │  base64 img ──→ [Vision Encoder] ──→ image_embeds                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │ completion + logprobs│                      │ image_embeds
        ▼                      │                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR                                      │
│  ┌────────────────┐                                                       │
│  │ verifiers env  │───▶ Extract: tokens, logprobs, images                 │
│  │ (OAI messages) │              │                                        │
│  └────────────────┘              │                                        │
│                                  ▼                                        │
│                         ┌──────────────────┐                              │
│                         │ Call /encode_images                             │
│                         │ on vLLM server   │                              │
│                         └────────┬─────────┘                              │
│                                  │                                        │
│                                  ▼                                        │
│                         ┌──────────────────┐                              │
│                         │ TrainingSample   │                              │
│                         │ + token_ids      │                              │
│                         │ + image_embeds   │◀──── uniform shape!          │
│                         └────────┬─────────┘                              │
└──────────────────────────────────┼────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            PACKER                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ token_ids ──→ [Embed Layer] ──→ text_embeds ──┐                  │  │
│  │                                                ├──→ [Pack] ──→ inputs_embeds
│  │ image_embeds ─────────────────────────────────┘                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┬────────────┘
                                                             │
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            TRAINER                                       │
│  ┌──────────────────┐    ┌─────────────────────────────────────────┐   │
│  │ MicroBatch       │───▶│ Language Model + LoRA                   │   │
│  │ - inputs_embeds  │    │ (NO vision encoder!)                    │   │
│  │ - position_ids   │    │                                         │   │
│  │ - loss_mask      │    │ forward(inputs_embeds) → logits         │   │
│  └──────────────────┘    └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: vLLM already loads the vision encoder for inference. We add one endpoint
to expose it, avoiding duplicate model loading.

---

### E. Key Implementation Decisions

1. **Freeze vision encoder** - confirmed by PM, enables simpler architecture
2. **Pre-compute image_embeds in orchestrator** - enables packing, reduces trainer memory
3. **Pack with embeddings** - text_embeds + image_embeds have uniform shape, standard packing works
4. **LoRA on language model only** - no vision encoder in trainer at all
5. **Pass inputs_embeds to trainer** - skip embedding layer, forward directly through LM
6. **Images pay in tokens** - no explicit image count limit, just seq_len limit
7. **Multi-image works naturally** - same approach, just more embeddings to concatenate

### F. Adding Vision Training Later

If we need to train the vision encoder in the future:
1. Move vision encoder back to trainer
2. Pass `pixel_values` instead of `image_embeds`
3. Either disable packing for vision samples OR
4. Implement variable-shape packing (more complex)

The current frozen approach is a subset - adding training later is straightforward.
