# d3

[d3-nano](https://huggingface.co/vllm-sr/d3-nano) (Decision 3.0, by the vLLM Semantic Router team) is Qwen3.5-2B with an answer-code readout. It answers one question per prompt: the prompt lists the question's options under short answer codes (`A`, `B`, ...), the full-attention layers read the prompt in both directions while the Gated DeltaNet layers stay causal, and a linear readout of 255 rows scores the codes at the prompt's last token. A question with n options keeps the first n scores. This folder fine-tunes the released model in full on [Typed Decisions Synth](../data/README.md), under the [contract](../README.md#the-contract) of the decision models.

## Files

| File | What |
|---|---|
| `encoder.py` | `D3Encoder`: every question is one d3 prompt (the state, the question, and its options under the answer codes; a score question lists its levels), each its own attention segment, read at its last token. The answer codes are derived from the tokenizer, as d3 derives them |
| `model.py` | `Qwen3_5D3`: prime-rl's Qwen3.5 plus `readout`, with the full-attention layers made non-causal. A question's logits are the readout's first rows at its prompt's last token, divided by the checkpoint's temperature |
| `prepare_checkpoint.py` | The checkpoint: the release under prime-rl's parameter names, with the readout and a `d3` entry in `config.json` (attention mode, temperature, readout rows) |
| `train.toml`, `smoke.toml` | The full fine-tuning run, and a 30-step mechanics check composed after it |

## Run d3

From the repository root, on one GPU. The data, the loss, the checkpoint layout and both configs are checked on CPU; neither config has run on a GPU yet, so memory use, the hyperparameters and the non-causal attention in prime-rl's flash-attention kernels are unmeasured. Prepare the checkpoint once, and the records as [their README](../data/README.md) says (they go to `outputs/typed-decisions-synth-system-one`):

```bash
uv run hf download vllm-sr/d3-nano --revision 462efc366229ea268cf4eed3738b3fc4415e4817 --local-dir outputs/d3/d3-nano
uv run python examples/decision_models/d3/prepare_checkpoint.py outputs/d3/d3-nano outputs/d3/d3-nano-prime-rl
```

Then the mechanics check, and the run. `PYTHONPATH=examples` makes `decision_models` importable in the trainer:

```bash
PYTHONPATH=examples uv run sft @ examples/decision_models/d3/train.toml @ examples/decision_models/d3/smoke.toml
PYTHONPATH=examples uv run sft @ examples/decision_models/d3/train.toml
```

The run is 550 steps: one pass over the 6,682 training records (23,319 questions) takes 543, and the last 7 begin a second pass. It validates on the 732 validation records before the first step and every 50 steps, the last time on the final step. The first validation measures the released model. The mechanics check keeps Qwen3.5's first four layers, of which the fourth is a full-attention layer. A run without a Prime API key needs `--no-monitors.prime`.

## Differences from d3

d3 published no training recipe, so the recipe is this example's:

- The whole language model and the readout train, at one learning rate (1e-5, untuned, cosine schedule with warmup), with AdamW, weight decay 0.01 and gradient clipping at 1.0. prime-rl keeps the vision encoder frozen, since the data is text.
- The loss is soft cross-entropy against the dataset's soft labels, without a Brier term.
- A step is two packed rows of up to 8,192 tokens each: 12.3 records, 43 prompts on average.

And the model runs as prime-rl runs it, not as d3's own runtime does:

- The readout computes in bf16 like the rest of the model, weights included, and its logits are then upcast. d3's runtime computes it in fp32, from fp32 weights. The difference grows with the norm of the hidden states and is unmeasured on real ones.
- The prompts of a record are packed back to back, each its own attention segment, in place of d3's batch of padded prompts. The full-attention layers run prime-rl's flash attention; d3 runs transformers' Qwen3.5 with SDPA.
- A record longer than 8,192 tokens is skipped. No record of this dataset is; the longest has 5,341 tokens.
- A score question lists every level, also when two levels have the same text, which d3's runtime refuses.
- Text only. d3-nano also reads images and videos.

The prompts, their tokens, the answer codes, the attention pattern, the readout and its temperature are d3's.

## Third-party notice

The prompt format and the answer codes in `encoder.py` are taken from `d3_format.py` of https://huggingface.co/vllm-sr/d3-nano at revision `462efc366229ea268cf4eed3738b3fc4415e4817`, rewritten compactly for System One questions, and `prepare_checkpoint.py` rewrites that release's weights and `config.json`. d3-nano is licensed under the Apache License, Version 2.0, which is also this repository's licence ([LICENSE](../../../LICENSE)). Its `NOTICE`:

```text
d3-nano (Decision 3.0)
Copyright 2026 vLLM Semantic Router Team

Built on Qwen/Qwen3.5-2B (Apache-2.0).
```
