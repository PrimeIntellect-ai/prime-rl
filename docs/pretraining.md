# Pretraining

`uv run pretrain` trains a model from scratch on raw text. It has its own config (`PretrainConfig`) and training loop, built from the same model setup, FSDP/EP/CP, optimizer, checkpointing and monitors as the SFT trainer. It runs on a single node.

```bash
uv run pretrain @ examples/pretrain/llama3/17m.toml
```

## Model

`model.name` names a baseline HF model. The baseline provides the architecture class, the tokenizer and the vocabulary. `[model.arch]` overrides fields of its HF config to set the size:

```toml
[model]
name = "meta-llama/Llama-3.2-1B"

[model.arch]
hidden_size = 384
num_hidden_layers = 8
num_attention_heads = 6
num_key_value_heads = 2
head_dim = 64
intermediate_size = 1536
```

The launcher writes the resulting config and the tokenizer to `<run_dir>/model` and trains from there, so the trainer, the HF export (`tools/convert_dcp_to_bf16.py`) and vLLM all see an ordinary model directory. The trainer logs the non-embedding, embedding (input embedding and LM head) and total parameter counts at startup.

The [Llama 3 ladder](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/pretrain/llama3) covers 17M, 70M, 150M, 500M and 1B non-embedding parameters.

## Initialization

Pretraining sets `model.debug.random_init = true`: weights are not downloaded or loaded, and `random_init_mode` resolves to `init`. Set `model.debug.random_init = false` to continue pretraining from the baseline's weights.

| Field | Meaning |
|---|---|
| `model.init.scheme` | `normal`: every weight matrix and the embedding from N(0, std). `scaled`: additionally divides the std of the residual output projections (`o_proj`, `down_proj`) by sqrt(2 * num_layers). Norm weights start at 1, biases at 0. |
| `model.init.std` | Base standard deviation (default 0.02). |
| `model.init.seed` | Seed. Every rank initializes only its own shard, and the values depend only on the seed and the parameter name, not on the number of GPUs. |

For `sft` and `rl`, `random_init_mode` resolves to `skip`: the weights keep arbitrary memory contents, which is the fastest option for throughput benchmarks.

## Schedules

A schedule is a multiplier `f(step)`, where `step` counts the optimizer updates completed so far (0 for the first update). Update `step` uses `base * f(step)` for every param group, with `base` the group's own configured value, so ratios between param groups hold at every step. Each optimizer hyperparameter gets its own schedule under `[schedules.optim.<name>]`, keyed by its `optim` field name: `lr`, `weight_decay`, `betas1`, `betas2`, and `mu` for Muon. `[schedules.clip_norm]` scales `optim.max_norm`. Hyperparameters without a schedule stay constant. The default is a WSD schedule on `lr`; setting `schedules.optim` replaces it, so include `lr` when scheduling other hyperparameters.

```toml
[schedules.optim.lr]
type = "wsd"
warmup = 0.05       # fraction of max_steps, or a number of steps
decay = 0.2
end_ratio = 0.1
decay_shape = "1-sqrt"

[schedules.optim.betas2]
type = "cosine"
warmup = 0
end_ratio = 1.05
```

| `type` | Shape |
|---|---|
| `wsd` (default) | Linear warmup, stable at 1, then a `linear`, `cosine` or `1-sqrt` decay over the last `decay` updates. Defaults: `warmup = 0.1`, `decay = 0.2`. |
| `cosine` | Linear warmup, then cosine decay over all remaining updates. |
| `constant` | 1 at every step. |
| `custom` | `fn(step, total_steps=max_steps, **kwargs) -> float` imported from `import_path`, with `kwargs` from the config. |

The configured endpoints are exactly what the first and last updates use: `f(0) = start_ratio`, the warmup reaches 1 at its last update (`warmup - 1`; a warmup of 0 or 1 starts at 1), and the decay reaches `end_ratio` at the last update (`max_steps - 1`). Both ratios default to 0. Fractions resolve against `max_steps`, so fractional phases and decays need `max_steps`.

AdamW and Muon apply weight decay as `lr * weight_decay`, so a learning rate schedule already scales the decay.

The values are recomputed from the step, so checkpoints hold no schedule state. Resuming with `ckpt.skip_scheduler = true` restarts every schedule at the resumed step, spanning the remaining steps. Each scheduled value is logged as `optim/<name>` (the first param group's value) and its multiplier as `schedule/<name>`.

## Data

`data.type = "text"` (the default) reads raw documents from a HF dataset column (`data.text_column`, default `text`). Only the selected subsets are downloaded. Each document is tokenized with the tokenizer's BOS and a closing EOS. Documents are packed back to back into rows of `data.seq_len` tokens and continue into the next row, so no token is truncated or padded. A step trains on exactly `data.batch_size * data.seq_len` tokens.

| Field | Meaning |
|---|---|
| `data.document_masking` | Attention stays within each document and positions restart at every document start (default). If false, a row is one sequence. Attention never crosses a row. |
| `data.micro_batch_size` | Rows per forward pass. |
| `data.holdout_docs` | Hold the last N documents (in dataset order) out of training. With `[val]` set, they are evaluated as the `holdout` source. |
| `data.subsets`, `data.splits`, `data.probabilities` | Load and interleave several subsets/splits. |

`data.type = "fake"` trains on generated tokens, for throughput tests.

## Evals

Evals run after the optimizer update, so the metrics at step N describe the weights after N updates, the same weights as the step-N checkpoint. With `eval_on_start` (default), they also run before the first step (logged as step 0), and they always run at the last step. They require `model.cp = 1`.

**Perplexity (`[val]`).** Each `[[val.source]]` is a held-out text set. It is tokenized and packed once at startup (with document masking, up to `max_tokens`), so every eval sees the same tokens. Each source logs `val/{name}/loss` (nats per token), `val/{name}/ppl` and `val/{name}/bpb` (bits per UTF-8 byte, comparable across tokenizers).

```toml
[val]
interval = 500

[[val.source]]
name = "wikitext"
dataset = "Salesforce/wikitext"
subset = "wikitext-103-raw-v1"
split = "test"
```

