# Decider

[pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) (by Perplexity) is Qwen3.5-4B with an answer-code readout. It answers every question of a record in one prompt: after the state, each question lists its options under short answer codes (`A`, `B`, ...) and ends in a marker token, `<|fim_suffix|>`. The full-attention layers read the prompt in both directions while the Gated DeltaNet layers stay causal, and a linear readout of 255 rows scores the codes at each marker. A question with n options keeps the first n scores. This folder fine-tunes the released model in full on [Typed Decisions Synth](../data/README.md), under the [contract](../README.md#the-contract) of the decision models.

## Files

| File | What |
|---|---|
| `encoder.py` | `DeciderEncoder`: the whole record is one prompt and one attention segment (a system prompt, the state, then every question with its options under the answer codes and a marker), read at each question's marker. The answer codes come from the checkpoint's `decision_config.json` |
| `model.py` | `Qwen3_5Decider`: prime-rl's Qwen3.5 plus `readout`, with the full-attention layers made non-causal. A question's logits are the readout's first rows at its marker. Its conversion methods put the release's weights under prime-rl's parameter names |
| `train.toml`, `smoke.toml` | The full fine-tuning run, and a 30-step mechanics check composed after it |

## Run Decider

From the repository root, on one GPU. The data, the loss and both configs are checked on CPU; neither config has run on a GPU yet, so memory use and the hyperparameters are unmeasured, and the conversion of the release has run only on a small stand-in, not on the release's weights. Download the release, and prepare the records as [their README](../data/README.md) says (they go to `outputs/typed-decisions-synth-system-one`):

```bash
uv run hf download perplexity-ai/pplx-decider-flash-v1.2 --revision 9ff7cf9838e59df461c276b7699d56b244d2af58 --local-dir outputs/decider/pplx-decider-flash-v1.2
```

Then the mechanics check, and the run. `PYTHONPATH=examples` makes `decision_models` importable in the trainer:

```bash
PYTHONPATH=examples uv run sft @ examples/decision_models/decider/train.toml @ examples/decision_models/decider/smoke.toml
PYTHONPATH=examples uv run sft @ examples/decision_models/decider/train.toml
```

The run is one pass over the 6,682 training records (23,319 questions) in 240 steps. It validates on the 732 validation records before the first step and every 30 steps, the last step included. The first validation measures the released model. The mechanics check keeps Qwen3.5's first four layers, of which the fourth is a full-attention layer. A run without a Prime API key needs `--no-monitors.prime`.

`model.name` is the release. It holds the backbone as a bare `Qwen3_5Model` (`language_model.*`, `visual.*`) and the readout apart (`readout.safetensors`, key `weight`), and prime-rl loads every parameter by its name in `Qwen3_5Decider` (`model.language_model.*`, `model.visual.*`, `readout.weight`). So the first load converts the release: prime-rl writes a copy of every weight, under the names `Qwen3_5Decider`'s conversion methods give it, to `outputs/decider/pplx-decider-flash-v1.2/prime/` (or under `model.conversion_dir`). Every later load reads that copy, which prime-rl never rebuilds: delete it if the release's files change. Completing an interrupted download is such a change: a run on a download that lacks a backbone shard still writes the copy, without that shard's weights, and every load that needs them fails with `The checkpoint lacks [...]` until the copy is deleted.

The encoder reads `decision_config.json` from the directory the tokenizer is loaded from, which is the release (`model.name`) unless `[tokenizer]` names another.

A record's attention segment must hold the record alone: the full-attention layers are non-causal, so every token of the record would attend to padding in the same segment. prime-rl packs records back to back without padding, and refuses `pad_to_seq_len` for System One data.

## Differences from Decider

Perplexity published no training recipe, so the recipe is this example's:

- The whole language model and the readout train, at one learning rate (1e-5, untuned, cosine schedule with warmup), with AdamW, weight decay 0.01 and gradient clipping at 1.0. prime-rl keeps the vision encoder frozen, since the data is text.
- The loss is soft cross-entropy against the dataset's soft labels, without a Brier term, on the readout's logits as they are. Decider divides them by a temperature at inference (1.55, in `decision_config.json`), fitted to the released weights. Validation therefore reports the loss of uncalibrated logits.
- A step is eight packed rows of up to 2,048 tokens each: about 28 records and 98 questions. A row is short to keep the activations small. In fp32 the 4.2B trained parameters and their gradients take 31 GiB on the GPU; AdamW's two moments take another 31 GiB, which prime-rl keeps in host memory except during the optimizer step (`model.offload = "optimizer"`, its default). None of this is measured.

And the model runs as prime-rl runs it, not as Decider's own code does:

- The records are packed back to back, each its own attention segment, in place of Decider's batch of left-padded prompts. The full-attention layers run prime-rl's flash attention.
- A record longer than 2,048 tokens is skipped. Decider splits the questions of such a request over several prompts. No record of this dataset is that long; the longest has 1,349 tokens.
- Text only. Decider also reads images, and takes its tokens from a processor. For a text prompt the processor applies the chat template and calls the tokenizer, which is what the encoder does.

The prompt, its tokens, the answer codes, the attention pattern and the readout are Decider's.

## Third-party notice

The prompt format in `encoder.py` is taken from `source/src/pplx_decider/protocol.py` of https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2, rewritten compactly for System One questions and text-only records, and `model.py` converts that release's weights to prime-rl's parameter names. pplx-decider-flash-v1.2 is licensed under the Apache License, Version 2.0, which is also this repository's licence ([LICENSE](../../../LICENSE)). Its `NOTICE`:

```text
pplx-decider-flash-v1.2

Derived from Qwen/Qwen3.5-4B, revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, under Apache License 2.0.
```
