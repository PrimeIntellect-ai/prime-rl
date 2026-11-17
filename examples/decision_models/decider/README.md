# Decider

[pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) (by Perplexity) is Qwen3.5-4B with an answer-code readout. It answers every question of a request in one prompt: after the state, each question lists its options under short answer codes (`A`, `B`, ...) and ends in a marker token, `<|fim_suffix|>`. The full-attention layers read the prompt in both directions while the Gated DeltaNet layers stay causal, and a linear readout of 255 rows scores the codes at each marker. A question with n options keeps the first n scores. This folder fine-tunes the released model in full on [Typed Decisions Synth](../data/README.md).

## Native and example code

A Decider release has d3's layout, so prime-rl selects its d3 class for it, `D3ForSequenceClassification` ([Native Decision Models](../../../docs/training.md#native-decision-models)), which reads the release's weights and builds its non-causal attention and float32 readout. The rest is this folder's:

| File | What |
|---|---|
| `model.py` | `Decider`, the `[model.implementation]` class: d3's class reading `decision_config.json` format version 2 only, Decider's (it refuses a d3 release, format version 1, whose prompts are d3's), declaring `DeciderStrategy`, and an export with temperature 1 |
| `strategy.py` | `DeciderStrategy`, a subclass of prime-rl's `D3Strategy`. `render_request`: the whole request is one prompt and one attention segment (a system prompt, the state, then every question with its options under the answer codes and a marker), read at each question's marker. `option_logits`: the readout's first rows at each marker, not divided by the temperature. The answer codes are those of the release's `decision_config.json` |
| `train.toml`, `smoke.toml` | The full fine-tuning run, and a 30-step mechanics check composed after it |

## vLLM

vLLM 0.33 does not serve Decider: it routes the release to d3's class and refuses its format version 2. A run's [export](../../../docs/training.md#exporting-checkpoints) has d3's layout (`decision_config.json` and `readout.safetensors` beside the shards), which loads for training again with the same `[model.implementation]`. Decider's `joint_config.json` and `source/` are not written, so copy them from the release to use Decider's own code. The export's `decision_config.json` has temperature 1 in place of the release's 1.55, so Decider's code serves the logits as they were trained, uncalibrated; refit the temperature on held-out data to calibrate them.

## Run

Download the release, and prepare the records as [their README](../data/README.md) says (they go to `outputs/typed-decisions-synth-system-one`), then run as [Running a model](../README.md#running-a-model) says, with `PYTHONPATH=examples`:

```bash
uv run hf download perplexity-ai/pplx-decider-flash-v1.2 --revision 9ff7cf9838e59df461c276b7699d56b244d2af58 --local-dir outputs/decider/pplx-decider-flash-v1.2
```

The run is one pass over the 6,682 training records (23,319 questions) in 240 steps. It validates on the 732 validation records before the first step and every 30 steps, the last step included. The mechanics check keeps Qwen3.5's first four layers, of which the fourth is a full-attention layer. Only the release's configuration and tokenizer have been loaded on CPU, not its weights.

A record's attention segment must hold the record alone: the full-attention layers are non-causal, so every token of the record would attend to anything else in the same segment. prime-rl gives each record segments of its own, and the padding of `pad_to_seq_len` a segment of its own.

## Differences from Decider

Perplexity published no training recipe, so the recipe is this example's:

- The whole language model and the readout train, at one learning rate (1e-5, untuned, cosine schedule with warmup), with AdamW, weight decay 0.01 and gradient clipping at 1.0. prime-rl keeps the vision encoder frozen, since the data is text.
- The loss is soft cross-entropy against the dataset's soft labels, without a Brier term, on the readout's logits as they are. Decider divides them by a temperature at inference (1.55, in `decision_config.json`), a calibration fitted to the released weights, which the strategy leaves out of the logits it trains. This departs on purpose from the `option_logits` contract, whose logits include a checkpoint's own temperature, until the export, whose temperature is 1. Validation therefore reports the loss of uncalibrated logits.
- A step is eight packed rows of up to 2,048 tokens each: about 28 records and 98 questions. A row is short to keep the activations small. In fp32 the 4.2B trained parameters and their gradients take 31 GiB on the GPU; AdamW's two moments take another 31 GiB, which prime-rl keeps in host memory except during the optimizer step (`model.offload = "optimizer"`, its default). None of this is measured.

And the model runs as prime-rl runs it, not as Decider's own code does:

- The records are packed back to back, each its own attention segment, in place of Decider's batch of left-padded prompts. The full-attention layers run prime-rl's flash attention.
- A record longer than 2,048 tokens is skipped. Decider splits the questions of such a request over several prompts. No record of this dataset is that long; the longest has 1,349 tokens.
- Text only. Decider also reads images, and takes its tokens from a processor. For a text prompt the processor applies the chat template and calls the tokenizer, which is what the strategy does.

The prompt, its tokens, the answer codes, the attention pattern and the readout are Decider's.

## Third-party notice

The prompt format in `strategy.py` is taken from `source/src/pplx_decider/protocol.py` of https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2, rewritten compactly for System One questions and text-only records. pplx-decider-flash-v1.2 is licensed under the Apache License, Version 2.0, which is also this repository's licence ([LICENSE](../../../LICENSE)). Its `NOTICE`:

```text
pplx-decider-flash-v1.2

Derived from Qwen/Qwen3.5-4B, revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, under Apache License 2.0.
```
