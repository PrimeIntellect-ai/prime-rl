# Pretraining

`uv run pretrain` trains a model from scratch on raw text. It has its own config (`PretrainConfig`) and training loop, built from the same model setup, FSDP/EP/CP, optimizer, scheduler, checkpointing and monitors as the SFT trainer. It runs locally on one node or, with `[slurm]`, as a SLURM job on one or more nodes (see [Multi-node](#multi-node)).

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
| `model.init.scheme` | `normal`: every weight matrix and the embedding from N(0, std). `scaled`: additionally divides the std of the residual output projections (`o_proj`, `down_proj`) by sqrt(2 * num_layers). `completep`: CompleteP, see below. Norm weights start at 1, biases at 0. |
| `model.init.std` | Base standard deviation (default 0.02). |
| `model.init.truncate` | Draw from the normal truncated at +-3 std. |
| `model.init.seed` | Seed. Every rank initializes only its own shard, and the values depend only on the seed and the parameter name, not on the number of GPUs. |

### CompleteP

`model.init.scheme = "completep"` trains with [CompleteP](https://arxiv.org/abs/2505.01618) (muP extended to depth, alpha = 1), so that hyperparameters tuned on a small base shape carry over to wider and deeper models. `optim.lr`, `optim.eps`, `optim.weight_decay` and `model.init.std` are the base-shape values, and the shape is set by `model.init.base_hidden_size` (default 256) and `model.init.base_num_layers` (default 2). With `m_N = hidden_size / base_hidden_size` and `m_L = num_layers / base_num_layers`:

| Parameters | Init std | Learning rate | Adam eps | Weight decay |
|---|---|---|---|---|
| Embedding | std | lr | eps / m_N | wd |
| Hidden matrices | std / sqrt(m_N) | lr / m_N | eps / (m_N m_L) | wd m_N |
| `o_proj`, `down_proj` | std / (sqrt(m_N) m_L) | lr / (m_N m_L) | eps / m_N | wd m_N m_L |
| LM head | std / m_N | lr / m_N | eps | wd m_N |
| Norm weights | 1 | lr | eps / m_N | 0 |

CompleteP scales each residual branch by 1/m_L and the logits by 1/m_N. These multipliers are folded into `o_proj`, `down_proj` and the LM head, so the model, its checkpoints and the exported weights are plain Llama.

With `optim.type = "muon"`, the hidden matrices and `o_proj`/`down_proj` are trained by Muon (`rms_norm` learning-rate adjustment), the rest by AdamW with the rules above. Muon's `rms_norm` update grows as sqrt(m_N), so its groups use:

| Parameters | Init std | Learning rate | Weight decay |
|---|---|---|---|
| Hidden matrices | std / sqrt(m_N) | lr / sqrt(m_N) | wd sqrt(m_N) |
| `o_proj`, `down_proj` | std / (sqrt(m_N) m_L) | lr / (sqrt(m_N) m_L) | wd sqrt(m_N) m_L |

CompleteP works with dense models, needs `model.offload = "none"`, and `scheduler.min_lr = 0` with the cosine scheduler. Keep `head_dim` and the ratios `intermediate_size / hidden_size` and `num_attention_heads / num_key_value_heads` fixed across shapes.

For `sft` and `rl`, `random_init_mode` resolves to `skip`: the weights keep arbitrary memory contents, which is the fastest option for throughput benchmarks.

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

## Multi-node

`uv run pretrain` takes the same `[deployment]` and `[slurm]` blocks as `uv run sft` (see [Scaling § SLURM](scaling.md#slurm)) and submits the same bundled templates. The launcher writes the model dir and downloads the data on the submitting node, then submits one job in which every node runs `torchrun` on the pretrain trainer:

```toml
output_dir = "/shared/outputs/pretrain"

[deployment]
type = "multi_node"
num_train_nodes = 4
gpus_per_node = 8
nodes_per_fsdp_group = 1  # optional: HSDP, shard within a node and replicate across nodes

[slurm]
partition = "all"
time = "04:00:00"
```

Each document goes to exactly one data stream, and there are `data_parallel_ranks * data.num_workers` streams, each packing its own documents into rows. The loss and gradient are normalized by the token count of the whole global batch, as on one GPU with gradient accumulation. The rows of a step depend on the number of streams: runs with different numbers of GPUs see the same data distribution but not the same batches, unless `data_parallel_ranks * data.num_workers` is the same (e.g. 1 GPU with 32 workers and 32 GPUs with 1 worker train on identical batches).
