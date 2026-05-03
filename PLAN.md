# MTP Joint Training Plan

## Scope

Implement MTP joint training for PrimeRL's custom Qwen3.5 MoE path.

The implementation is intentionally scoped to `qwen3_5_moe_text` models. Dense Qwen3.5 and Nemotron-H MTP support are deferred follow-ups. Nemotron conversion keeps the existing `mtp.*` drop behavior, but now documents that this is an explicit deferral instead of an accidental omission.

## Goals

- Preserve Qwen3.5 MoE MTP weights through HF -> PrimeRL -> HF conversion.
- Add an auxiliary MTP cross-entropy loss that trains only MTP layers.
- Keep the main policy trunk, embeddings, and LM head insulated from MTP loss gradients.
- Wire MTP loss through both SFT and RL training paths.
- Support MTP weight broadcast through NCCL for non-layer MTP checkpoint chunks.
- Reject unsupported combinations early: context parallel MTP, VLM MTP, non-custom model implementations, and quantized NCCL transfer with MTP.
- Add a vLLM `speculative_config` plumbing point for rollout-side speculative decoding.

## Design

### Configuration

`ModelConfig.mtp` controls MTP training:

- `enabled`: enables auxiliary MTP training.
- `loss_scale`: scales the auxiliary CE loss.
- `enable_rollout`: enables inference-side speculative decoding config wiring.
- `num_speculative_tokens`: overrides rollout speculation depth.

When MTP rollout is enabled, RL config injects a default vLLM speculative config unless the user already provided one explicitly.

### Shared Helpers

`src/prime_rl/trainer/mtp.py` owns model-agnostic helpers:

- `roll_tensor`: left-shifts tensors without crossing packed-sequence boundaries when `position_ids` are supplied.
- `mtp_masks_from_label_mask`: builds cumulative valid-token masks for MTP targets.
- `make_viewless_tensor_with_grad`: detaches trunk hidden states while allowing MTP-local gradients.
- `detached_lm_head_cross_entropy`: computes auxiliary CE through detached LM-head weights.

The detach behavior is load-bearing. MTP loss must not update the trunk, embedding table, or LM head.

### Qwen3.5 MoE Model

Qwen3.5 MoE gains optional `mtp_layers`, each composed of:

- embedding RMSNorm
- hidden-state RMSNorm
- `eh_proj`
- one full-attention `Qwen3_5MoeDecoderLayer`
- final norm

Each depth predicts the next future token from the previous trunk/MTP hidden state plus the rolled token embedding. The auxiliary logits use the shared LM head as a read-only function.

### Training

SFT and RL forward passes now pass label-aligned loss masks to the model. If the model returns `mtp_loss`, the training loss adds the scaled auxiliary term and logs MTP metrics.

RL keeps the policy loss path unchanged and uses shifted masks for the MTP objective.

### Conversion And Broadcast

Qwen3.5 MoE conversion maps HF top-level `mtp.*` keys to PrimeRL `mtp_layers.*` keys and back.

NCCL broadcast converts non-layer MTP chunks back to HF shape before sending to vLLM. Quantized NCCL transfer is rejected when MTP weights are present.

Filesystem broadcast already uses full-state conversion and does not need a separate MTP path.

### Deferred Work

- Dense Qwen3.5 MTP support, including official `Qwen/Qwen3.5-0.8B-Base`.
- Nemotron-H MTP model support.
- Context-parallel MTP training.
- Real rollout benchmark against vLLM speculative decoding.
- Model-specific vLLM method defaults such as Qwen's `qwen3_next_mtp`, if required by the installed vLLM version.

## Acceptance Criteria

Do not stop work before all of these are true for the implemented Qwen3.5 MoE scope:

- MTP is disabled by default and existing non-MTP model tests still pass.
- Enabling MTP on an unsupported model fails clearly during setup.
- Enabling MTP with CP, VLM, or quantized NCCL transfer fails clearly during config validation or broadcast setup.
- Qwen3.5 MoE MTP layers are registered before checkpoint loading, so DCP schema loading can request MTP weights.
- HF -> PrimeRL -> HF conversion preserves Qwen3.5 MoE `mtp.*` keys.
- Packed-sequence rolling does not cross sequence boundaries.
- MTP loss masks require both the intermediate token and target token to be valid.
- With a zero main loss, gradients flow to `mtp_layers.*` and not to trunk, embedding, or LM-head parameters.
- SFT and RL training paths can add and log MTP loss without changing the non-MTP path.
- NCCL preprocessing converts non-layer MTP checkpoint chunks back to HF key shape for vLLM.
- The focused GPU test suite passes on GPU 1.

## Verified

The focused GPU suite was run on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 uv run pytest \
  tests/unit/train/models/test_qwen3_5_moe_mtp.py \
  tests/unit/train/rl/test_nccl_broadcast.py::test_nccl_preprocess_converts_mtp_non_layer_chunk_to_hf_keys \
  tests/unit/train/models/test_qwen3_5_moe.py \
  tests/unit/train/test_model_forward.py \
  tests/unit/train/rl/test_fused_lm_head.py \
  -q
```

Result: 21 passed.
