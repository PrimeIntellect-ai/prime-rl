# d3

[d3-nano](https://huggingface.co/vllm-sr/d3-nano) (Decision 3.0, by the vLLM Semantic Router team) is Qwen3.5-2B with an answer-code readout. It answers one question per prompt: the prompt lists the question's options under short answer codes (`A`, `B`, ...), the full-attention layers read the prompt in both directions while the Gated DeltaNet layers stay causal, and a linear readout of 255 rows scores the codes at the prompt's last token. A question with n options keeps the first n scores. This folder fine-tunes the released model in full on [Typed Decisions Synth](../data/README.md).

## Native and example code

The model class and its strategy are prime-rl's ([Native Decision Models](../../../docs/training.md#native-decision-models)); this folder holds only the configs. `model.name` is the release as downloaded, which prime-rl builds as `D3ForSequenceClassification`, the class vLLM 0.33 serves d3 with, so the config has no `[model.implementation]`. The class declares the in-tree `D3Strategy`, which builds the prompts of vLLM 0.33's `/v1/systemone` and reads the readout as vLLM 0.33's `D3Strategy` does.

| File | What |
|---|---|
| `train.toml`, `smoke.toml` | The full fine-tuning run, and a 30-step mechanics check composed after it |

## vLLM

vLLM 0.33 serves d3. The prompts and the readout (in float32, divided by the temperature of `decision_config.json`, 1.0 for d3-nano) are vLLM's, and a run's [export](../../../docs/training.md#exporting-checkpoints) has the release's layout (`decision_config.json` and `readout.safetensors` beside the shards), which vLLM 0.33 serves. On the export of the full run, vLLM's probabilities and the trainer's differ by a median of about 0.001 per question, as much as vLLM's own differ under another batching. Serve it with the router off (`router = "None"` in the `uv run inference` config, or `--router None`), or send requests to the engine port: vllm-router re-sorts the JSON keys of a request, which reorders d3's options and state keys, so the served prompts differ from the trained ones.

## Run

Download the release, and prepare the records as [their README](../data/README.md) says (they go to `outputs/typed-decisions-synth-system-one`), then run as [Running a model](../README.md#running-a-model) says:

```bash
uv run hf download vllm-sr/d3-nano --revision 462efc366229ea268cf4eed3738b3fc4415e4817 --local-dir outputs/d3/d3-nano
```

The run is 550 steps: one pass over the 6,682 training records (23,319 questions) takes 543, and the last 7 begin a second pass. It validates on the 732 validation records before the first step and every 50 steps, the last time on the final step. The mechanics check keeps Qwen3.5's first four layers, of which the fourth is a full-attention layer.

On two H100s (`--deployment.num-train-gpus 2`), before d3's prompts and readout moved into prime-rl's strategy ([Running a model](../README.md#running-a-model)), the run took about 10 minutes, a step about 0.8 s, and 21.5 GiB per GPU, and the validation loss went from 0.9528 to 0.6737; on one GPU a step took about 1.5 s and 36.4 GiB. On two GPUs those losses covered 719 of the 732 validation records: validation stops when the first rank runs out of packed rows ([System One data](../../../docs/training.md#system-one-data)). On an H100, `attn = "auto"` runs the non-causal full-attention layers in FlashAttention 3.

## Differences from d3

d3 published no training recipe, so the recipe is this example's:

- The whole language model and the readout train, at one learning rate (1e-5, cosine schedule with warmup; 2e-5 for 30 steps also trained, while 1e-4 for 10 steps collapsed the model to near-uniform answers, validation loss 0.95 to 1.18), with AdamW, weight decay 0.01 and gradient clipping at 1.0. prime-rl keeps the vision encoder frozen, since the data is text.
- The loss is soft cross-entropy against the dataset's soft labels, without a Brier term.
- A step is two packed rows of up to 8,192 tokens each: 12.3 records, 43 prompts on average.

And the model runs as prime-rl runs it:

- The prompts of a record are packed back to back, each its own attention segment, in place of a batch of padded prompts. The full-attention layers run prime-rl's flash attention.
- A record whose prompts together are longer than 8,192 tokens is skipped. No record of this dataset is; the longest has 5,341 tokens.
- Text only. d3-nano also reads images and videos.
