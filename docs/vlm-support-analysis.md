# Vision Language Model (VLM) Support Analysis for Prime-RL

This document analyzes the changes required to support vision language models like [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) in the Prime-RL codebase for both SFT and RL training pipelines.

## Table of Contents

1. [Current Architecture Overview](#1-current-architecture-overview)
2. [Component Interactions](#2-component-interactions)
3. [VLM Architecture Requirements (Qwen3-VL)](#3-vlm-architecture-requirements-qwen3-vl)
4. [Core Changes Required](#4-core-changes-required)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Technical Considerations](#6-technical-considerations)

---

## 1. Current Architecture Overview

Prime-RL consists of three main components that work together:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                    │
│  - Loads examples from dataset                                              │
│  - Manages trajectory buffer for rollouts                                   │
│  - Communicates with inference server via OpenAI-compatible API             │
│  - Computes advantages and prepares training batches                        │
│  - Tokenizes prompts using AutoTokenizer                                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ TrainingBatch (TrainingSamples)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               TRAINER                                        │
│  - Receives micro batches from packer                                       │
│  - Runs forward/backward passes with FSDP                                   │
│  - Supports RL (GRPO/PPO) and SFT training                                  │
│  - Broadcasts updated weights to inference                                  │
│  - Uses AutoModelForCausalLM (text-only)                                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ Weight updates (NCCL/Filesystem)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE SERVER                                   │
│  - vLLM-based serving with OpenAI-compatible API                            │
│  - Custom /v1/chat/completions/tokens endpoint                              │
│  - Accepts pre-tokenized inputs for consistency                             │
│  - Returns logprobs and token IDs                                           │
│  - Receives weight updates during training                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Key Data Structures

**TrainingSample** (`transport/types.py`):
```python
class TrainingSample(msgspec.Struct):
    prompt_ids: list[int]           # Tokenized prompt
    prompt_mask: list[bool]         # Attention mask for prompt
    completion_ids: list[int]       # Tokenized completion
    completion_mask: list[bool]     # Loss mask for completion
    completion_logprobs: list[float] # Logprobs from generation
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None
```

**MicroBatch** (`transport/types.py`):
```python
class MicroBatch(msgspec.Struct):
    input_ids: list[int]           # Concatenated prompt + completion
    loss_mask: list[bool]          # Which tokens contribute to loss
    advantages: list[float]        # Per-token advantages
    inference_logprobs: list[float]
    position_ids: list[int]
    temperature: float
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
```

**TensorMicroBatch** (`trainer/rl/data.py`):
```python
class TensorMicroBatch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    teacher_logprobs: Float[Tensor, "batch seq"] | None
    loss_mask: Bool[Tensor, "batch seq"]
    temperature: float
    lora_num_tokens: Int[Tensor, "n_loras"]
```

### 1.2 Current Model Loading

In `trainer/model.py`:
```python
def get_model(config: ModelConfig, device, dtype):
    model_config = AutoConfig.from_pretrained(config.name, ...)

    match impl_to_use:
        case "hf":
            model_cls = AutoModelForCausalLM  # Text-only!
        case "liger_kernel":
            model_cls = AutoLigerKernelForCausalLM
        case "custom":
            model_cls = AutoModelForCausalLMPrimeRL

    model = model_cls.from_config(model_config, ...)
    return model
```

The `forward` function signature:
```python
def forward(
    model: nn.Module,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    labels: Int[Tensor, "batch seq"] | None = None,
    temperature: float | None = None,
) -> PrimeLmOutput:
```

---

## 2. Component Interactions

### 2.1 RL Training Flow

```
1. ORCHESTRATOR
   └─► Loads examples with prompts from dataset
   └─► Sends prompts to inference server
   └─► Receives completions + logprobs
   └─► Runs reward verification (verifiers library)
   └─► Computes advantages
   └─► Creates TrainingSample objects
   └─► Packs into TrainingBatch

2. PACKER (part of trainer)
   └─► Receives TrainingBatch
   └─► Prepares samples (concat prompt+completion)
   └─► Packs multiple samples (bin packing)
   └─► Creates MicroBatch objects

3. TRAINER
   └─► Receives MicroBatch
   └─► Converts to TensorMicroBatch (GPU tensors)
   └─► Forward pass: model(input_ids, position_ids)
   └─► Compute loss with advantages
   └─► Backward pass
   └─► Optimizer step
   └─► Broadcast weights to inference

4. INFERENCE SERVER
   └─► Receives weight update
   └─► Updates model parameters in-place
   └─► Ready for next generation
```

### 2.2 SFT Training Flow

```
1. DATASET (trainer/sft/data.py)
   └─► SFTDataset loads examples with prompt/completion
   └─► Applies chat template via tokenizer
   └─► Builds loss mask (train on assistant only)
   └─► Creates input_ids, target_ids, loss_mask, position_ids

2. DATALOADER
   └─► StackDataset or CatDataset for packing
   └─► Collates into batches

3. TRAINER (trainer/sft/train.py)
   └─► Forward: model(input_ids, position_ids)
   └─► CrossEntropyLoss on logits vs target_ids
   └─► Masked by loss_mask
   └─► Backward + optimizer step
```

### 2.3 Tokenization Points

| Component | Tokenization Method | Current Handling |
|-----------|-------------------|------------------|
| Orchestrator | `AutoTokenizer.from_pretrained()` | Text-only tokenization |
| SFT Dataset | `tokenizer.apply_chat_template()` | Text messages, supports tool calls |
| Inference Server | vLLM internal | Text tokenization via model's tokenizer |
| Trainer | Receives pre-tokenized data | No tokenization, just tensor conversion |

---

## 3. VLM Architecture Requirements (Qwen3-VL)

### 3.1 Model Architecture

Qwen3-VL uses a three-component architecture:

```
┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Vision Encoder  │────►│    Projector    │────►│      LLM         │
│  (ViT-based)     │     │  (MLP Adapter)  │     │  (Qwen3 Decoder) │
└──────────────────┘     └─────────────────┘     └──────────────────┘
       ▲                                                  ▲
       │                                                  │
   Images/Video                                      Text tokens
```

**Key innovations in Qwen3-VL**:
- Enhanced M-RoPE with interleaved layout for spatial-temporal modeling
- DeepStack integration for multi-level ViT features
- Text-based timestamp alignment for video understanding

### 3.2 Required Inputs

The model's forward function requires:

```python
def forward(
    self,
    input_ids: torch.LongTensor,           # Text token IDs
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    pixel_values: torch.Tensor,             # NEW: Image features
    image_grid_thw: torch.LongTensor,       # NEW: (num_images, 3) - temporal, height, width
    pixel_values_videos: torch.FloatTensor, # NEW: Video features (optional)
    video_grid_thw: torch.LongTensor,       # NEW: Video grid info (optional)
    ...
)
```

**`image_grid_thw`** shape: `(num_images, 3)` representing the temporal (T), height (H), and width (W) of the feature grid for each image in the sequence.

### 3.3 Processing Pipeline

```python
from transformers import Qwen3VLProcessor

processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Message format with images
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "Describe this image"}
    ]}
]

# Process returns both text tokens AND image tensors
inputs = processor(
    text=text,
    images=images,
    return_tensors="pt"
)
# Returns: input_ids, attention_mask, pixel_values, image_grid_thw
```

### 3.4 Vision Token Embedding

1. Images are processed by the vision encoder → feature grid
2. Feature grid is projected to LLM hidden dimension
3. Special `<image>` placeholder tokens in input_ids are replaced with vision features
4. Combined sequence is processed by the LLM decoder

---

## 4. Core Changes Required

### 4.1 Model Loading Changes (`trainer/model.py`)

**Current**:
```python
model_cls = AutoModelForCausalLM
```

**Required**:
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

def get_model(config: ModelConfig, device, dtype):
    if config.is_vlm:  # New config flag
        model_cls = AutoModelForVision2Seq  # Or Qwen3VLForConditionalGeneration
        # Also need to load processor for image preprocessing
        processor = AutoProcessor.from_pretrained(config.name)
    else:
        model_cls = AutoModelForCausalLM
```

**New model base class** for custom VLM implementations:
```python
class PreTrainedModelVLMPrimeRL(PreTrainedModelPrimeRL):
    """Base class for VLM models with vision encoder."""

    def get_vision_encoder(self) -> nn.Module:
        """Return the vision encoder component."""
        raise NotImplementedError

    def get_projector(self) -> nn.Module:
        """Return the vision-to-text projector."""
        raise NotImplementedError
```

### 4.2 Forward Function Changes (`trainer/model.py`)

**Current**:
```python
def forward(
    model: nn.Module,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    labels: Int[Tensor, "batch seq"] | None = None,
    temperature: float | None = None,
) -> PrimeLmOutput:
    out = model(input_ids=input_ids, position_ids=position_ids, ...)
```

**Required**:
```python
def forward(
    model: nn.Module,
    input_ids: Int[Tensor, "batch seq"],
    position_ids: Int[Tensor, "batch seq"],
    labels: Int[Tensor, "batch seq"] | None = None,
    temperature: float | None = None,
    # NEW VLM inputs
    pixel_values: Float[Tensor, "batch channels height width"] | None = None,
    image_grid_thw: Int[Tensor, "num_images 3"] | None = None,
    pixel_values_videos: Float[Tensor, "..."] | None = None,
    video_grid_thw: Int[Tensor, "num_videos 3"] | None = None,
) -> PrimeLmOutput:
    if pixel_values is not None:
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            ...
        )
    else:
        out = model(input_ids=input_ids, position_ids=position_ids, ...)
```

### 4.3 Data Structure Changes

**TrainingSample** (`transport/types.py`):
```python
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None
    # NEW VLM fields
    image_data: bytes | None = None  # Serialized image tensor or path
    image_grid_thw: list[list[int]] | None = None  # Per-image grid info
    num_image_tokens: int | None = None  # How many tokens images occupy
```

**MicroBatch** (`transport/types.py`):
```python
class MicroBatch(msgspec.Struct, ...):
    # ... existing fields ...
    # NEW VLM fields
    pixel_values: list[float] | None = None  # Flattened image tensor
    pixel_values_shape: list[int] | None = None  # Shape for reconstruction
    image_grid_thw: list[int] | None = None  # Flattened grid info
    image_positions: list[int] | None = None  # Token positions with images
```

**TensorMicroBatch** (`trainer/rl/data.py`):
```python
class TensorMicroBatch(TypedDict):
    # ... existing fields ...
    # NEW VLM fields
    pixel_values: Float[Tensor, "batch_images channels height width"] | None
    image_grid_thw: Int[Tensor, "batch_images 3"] | None
    image_token_mask: Bool[Tensor, "batch seq"] | None  # Which positions are image tokens
```

### 4.4 SFT Dataset Changes (`trainer/sft/data.py`)

**Current `_process` method**:
```python
def _process(self, example: dict) -> Sample | None:
    input_ids = self.tokenizer.apply_chat_template(
        prompt + completion,
        tools=tools,
    )
    # ... build loss_mask ...
    return {"input_ids": input_ids, "target_ids": target_ids, ...}
```

**Required changes**:
```python
def _process(self, example: dict) -> Sample | None:
    # Extract images from multimodal content
    images = self._extract_images(example)

    if images and self.processor is not None:
        # Use processor for multimodal
        inputs = self.processor(
            text=self._build_text_from_messages(prompt + completion),
            images=images,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze().tolist()
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
    else:
        # Text-only fallback
        input_ids = self.tokenizer.apply_chat_template(...)
        pixel_values = None
        image_grid_thw = None

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "pixel_values": pixel_values,  # NEW
        "image_grid_thw": image_grid_thw,  # NEW
    }

def _extract_images(self, example: dict) -> list[Image]:
    """Extract images from multimodal message content."""
    images = []
    for msg in example.get("prompt", []) + example.get("completion", []):
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    images.append(self._load_image(part["image"]))
    return images
```

### 4.5 SFT Training Loop Changes (`trainer/sft/train.py`)

**Current**:
```python
for micro_step in range(grad_accum_steps):
    micro_batch = next(dataiter)
    input_ids = micro_batch["input_ids"].to("cuda")
    position_ids = micro_batch["position_ids"].to("cuda")
    # ...
    out = forward(model, input_ids, position_ids)
```

**Required**:
```python
for micro_step in range(grad_accum_steps):
    micro_batch = next(dataiter)
    input_ids = micro_batch["input_ids"].to("cuda")
    position_ids = micro_batch["position_ids"].to("cuda")

    # NEW: Handle vision inputs
    pixel_values = micro_batch.get("pixel_values")
    image_grid_thw = micro_batch.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to("cuda")
        image_grid_thw = image_grid_thw.to("cuda")

    out = forward(
        model, input_ids, position_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
```

### 4.6 RL Training Loop Changes (`trainer/rl/train.py`)

Similar changes as SFT, but also need to handle vision data in:
- `DataLoader.get_batch()` - deserialize image tensors
- The micro-batch processing loop
- The `forward()` call

### 4.7 Orchestrator Changes (`orchestrator/orchestrator.py`)

**Current tokenizer setup**:
```python
tokenizer = AutoTokenizer.from_pretrained(config.model.name, ...)
```

**Required**:
```python
tokenizer = AutoTokenizer.from_pretrained(config.model.name, ...)

# NEW: Also load processor for VLM
if config.model.is_vlm:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config.model.name, ...)
```

**Example handling** in buffer and scheduler needs to:
1. Store image paths/URLs with examples
2. Load images lazily when creating rollout inputs
3. Pass image data to verifiers library

### 4.8 Inference Server Changes (`inference/vllm/server.py`)

vLLM already supports multimodal models. Key changes:

1. **Model loading**: Use vLLM's multimodal model loader
2. **Request handling**: Accept image URLs/base64 in messages
3. **Token endpoint**: Handle mixed image+text token sequences

```python
# Example multimodal request to vLLM
{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            {"type": "text", "text": "Describe this image"}
        ]}
    ]
}
```

### 4.9 FSDP Sharding Changes (`trainer/model.py:setup_fsdp`)

**Current**:
```python
def setup_fsdp(model, config, parallel_dims):
    for transformer_block in model.model.layers:
        fully_shard(transformer_block, ...)
    fully_shard(model.model.embed_tokens, ...)
    fully_shard([model.lm_head, model.model.norm], ...)
```

**Required for VLM**:
```python
def setup_fsdp(model, config, parallel_dims):
    # Shard vision encoder separately
    if hasattr(model, "vision_tower") or hasattr(model, "visual"):
        vision_encoder = getattr(model, "vision_tower", None) or model.visual
        for vision_block in vision_encoder.blocks:  # ViT blocks
            fully_shard(vision_block, ...)

    # Shard projector
    if hasattr(model, "multi_modal_projector"):
        fully_shard(model.multi_modal_projector, ...)

    # Shard LLM layers (existing)
    for transformer_block in model.model.layers:
        fully_shard(transformer_block, ...)
```

### 4.10 Configuration Changes

**New config fields** (`trainer/config.py`):
```python
class ModelConfig(BaseConfig):
    # ... existing fields ...
    is_vlm: bool = False
    freeze_vision_encoder: bool = True  # Common for VLM fine-tuning
    image_size: int | None = None
    max_image_tokens: int = 1280  # Qwen3-VL default
```

**Orchestrator config** (`orchestrator/config.py`):
```python
class DataConfig(BaseConfig):
    # ... existing fields ...
    image_dir: Path | None = None  # Base directory for images
    load_images_lazily: bool = True
    max_image_size: int = 1024  # Resize large images
```

---

## 5. Implementation Roadmap

### Phase 1: SFT Pipeline (Foundation)

1. **Model loading for VLM**
   - Add `AutoModelForVision2Seq` support
   - Add processor loading alongside tokenizer
   - Update `forward()` signature

2. **SFT data processing**
   - Extend `SFTDataset._process()` for images
   - Add image extraction from multimodal messages
   - Handle processor-based tokenization+image processing

3. **SFT training loop**
   - Update batch handling for vision inputs
   - Add vision-specific FSDP sharding
   - Handle variable-sized image batches

### Phase 2: RL Pipeline

4. **Data structures**
   - Extend `TrainingSample` and `MicroBatch`
   - Update serialization for image data
   - Handle image packing in batch preparation

5. **Orchestrator changes**
   - Load and pass images with examples
   - Update verifiers integration for multimodal
   - Handle image data in trajectory buffer

6. **Trainer changes**
   - Update `DataLoader` for vision data
   - Extend RL training loop for VLM inputs

### Phase 3: Inference Integration

7. **vLLM multimodal support**
   - Enable multimodal models in server config
   - Handle image URLs/base64 in requests
   - Ensure weight updates work for VLM

8. **Weight broadcasting**
   - Handle vision encoder weights
   - Coordinate projector weight updates
   - Test end-to-end RL loop

### Phase 4: Optimization

9. **Memory optimization**
   - Lazy image loading
   - Image caching strategies
   - Gradient checkpointing for vision encoder

10. **Performance optimization**
    - Batch image preprocessing
    - Efficient image serialization
    - Parallel image loading

---

## 6. Technical Considerations

### 6.1 Memory Management

VLMs require significantly more memory due to:
- Vision encoder parameters (~300M for ViT-L)
- Image feature tensors (can be 4-10x text tokens)
- Projector parameters

**Strategies**:
- Freeze vision encoder (common practice)
- Use gradient checkpointing for vision tower
- Lazy image loading
- Image resolution management

### 6.2 Batch Handling

Images have variable sizes, creating challenges:
- Different numbers of vision tokens per image
- Grid sizes vary with image aspect ratio
- Video adds temporal dimension

**Solutions**:
- Pad to maximum vision tokens in batch
- Use mask to ignore padding
- Consider dynamic batching by image count

### 6.3 Serialization

Images are large and require special handling:
- Store paths/URLs instead of raw bytes when possible
- Use efficient formats (WebP, JPEG)
- Consider tensor compression for transport

### 6.4 Tokenization Consistency

Critical for RL training:
- Orchestrator and inference must produce identical tokenization
- Image token positions must match exactly
- Use processor consistently across components

### 6.5 Loss Masking for VLM

Typically don't train on:
- Image placeholder tokens (model learns to use features, not predict them)
- System/user messages
- Only train on assistant responses about images

```python
# Example loss mask for VLM
# [sys_tokens..., user_tokens..., <img>, <img>, ..., assistant_tokens...]
# [  False...,      False...,     False, False, ...,    True...        ]
```

### 6.6 Position IDs for Vision Tokens

Different VLMs handle this differently:
- Some use contiguous position IDs
- Qwen3-VL uses M-RoPE with interleaved layout
- Must respect model's position embedding scheme

---

## References

- [Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL Technical Report (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [HuggingFace Transformers VLM Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl)
- [vLLM Multimodal Support](https://docs.vllm.ai/en/latest/models/vlm.html)
