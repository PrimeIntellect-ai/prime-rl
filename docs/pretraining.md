# Pretraining

`uv run pretrain` trains a model from scratch on pre-tokenized token shards. It has its own config (`PretrainConfig`) and training loop, built from the same model setup, FSDP/EP/CP, optimizer, checkpointing and monitors as the SFT trainer. It runs on a single node.

```bash
uv run pretrain @ examples/pretrain/glm4_moe.toml
```

## Model

`[model.arch]` is the whole architecture: a `model_type` from `prime_rl.trainer.models` and the fields of its config class. Unknown fields are rejected. The vocabulary size (`len(tokenizer)` rounded up to a multiple of 128) and the BOS, EOS and pad ids come from the tokenizer, so they cannot disagree with it.

```toml
[model.arch]
model_type = "glm4_moe"
hidden_size = 512
num_hidden_layers = 8
n_routed_experts = 64
num_experts_per_tok = 6

[tokenizer]
name = "tokenizers/deepseek-v4-flash-base"
```

The launcher writes the config and the tokenizer to `<run_dir>/model` and trains from there, so the trainer, the HF export (`tools/convert_dcp_to_bf16.py`) and vLLM all see an ordinary model directory. The trainer logs the non-embedding, embedding (input embedding and LM head) and total parameter counts at startup.

## Initialization

Weights start from a random init, unless the run resumes from a checkpoint.

| Field | Meaning |
|---|---|
| `model.init.scheme` | `normal`: every weight matrix and the embedding from N(0, std). `scaled`: additionally divides the std of the residual output projections (`o_proj`, `down_proj`) by sqrt(2 * num_layers). Norm weights start at 1, biases at 0. |
| `model.init.std` | Base standard deviation (default 0.02). |
| `model.init.seed` | Seed. Every rank initializes only its own shard, and the values depend only on the seed and the parameter name, not on the number of GPUs. |

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
| `wsd` (default) | Linear warmup, stable at 1, then a `linear`, `cosine` or `1-sqrt` decay over the last `decay` updates. Defaults: `warmup = 0.1`, `decay = 0.9`, i.e. a linear decay to 0 over all updates after the warmup. |
| `cosine` | Linear warmup, then cosine decay over all remaining updates. |
| `constant` | 1 at every step. |
| `custom` | `fn(step, total_steps=max_steps, **kwargs) -> float` imported from `import_path`, with `kwargs` from the config. |

The configured endpoints are exactly what the first and last updates use: `f(0) = start_ratio`, the warmup reaches 1 at its last update (`warmup - 1`; a warmup of 0 or 1 starts at 1), and the decay reaches `end_ratio` at the last update (`max_steps - 1`). Both ratios default to 0. Fractions resolve against `max_steps`, so fractional phases and decays need `max_steps`.

AdamW and Muon apply weight decay as `lr * weight_decay`, so a learning rate schedule already scales the decay.

The values are recomputed from the step, so checkpoints hold no schedule state. Resuming with `ckpt.skip_scheduler = true` restarts every schedule at the resumed step, spanning the remaining steps. Each scheduled value is logged as `optim/<name>` (the first param group's value) and its multiplier as `schedule/<name>`.

## Data

`data.type = "tokens"` reads [datatrove](https://github.com/huggingface/datatrove) `DocumentTokenizer` output: per source, a folder of `*.ds` token files with their `.ds.index` (document ends) and `.ds.metadata`. The files of a source are read in name order as one stream; documents continue across rows, and attention restarts at every document and every row. Every document must end with the tokenizer's EOS; the trainer checks the first document of every source, which catches shards made with another tokenizer.

Each row comes from one source, drawn with the weights of the current mixture phase. Phases start at fractions of `max_steps`. At startup the trainer computes how many passes over each source the run makes and fails above `data.max_epochs`.

```toml
[data]
type = "tokens"
seq_len = 4096
batch_tokens = 262144       # per optimizer step, over all GPUs
micro_batch_tokens = 65536  # per forward pass on one GPU
max_epochs = 3

[data.sources]
web = { path = "data/web" }
code = { path = "data/code" }

[[data.phases]]
weights = { web = 0.8, code = 0.2 }

[[data.phases]]
start = 0.9
weights = { web = 0.5, code = 0.5 }
```

Batch sizes are in tokens, so changing `seq_len` keeps the tokens per step and the number of steps. The resume state is the global row counter and the tokens read per source, the same on every rank. `data.type = "fake"` trains on random tokens, for smoke and throughput tests.

## Evals

`[val]` computes the held-out loss (nats per token), perplexity and bits per UTF-8 byte (comparable across tokenizers) of every source in `val.sources`, through the same loss code as training, so it also works with context parallelism. Each eval reads the same leading windows of every source: as many as fill whole micro-batches on every data-parallel rank. Evals run after the optimizer update, so the metrics at step N describe the same weights as the step-N checkpoint. With `eval_on_start` (default), they also run before the first step (logged as step 0), and they always run at the last step.

```toml
[val]
interval = 500

[val.sources]
web = { path = "data/web-val" }
```
