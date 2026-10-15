# Pretraining

`uv run pretrain` trains a model from scratch on pre-tokenized token shards. It has its own config (`PretrainConfig`) and training loop, built from the same model setup, FSDP/EP/CP, optimizer, checkpointing and monitors as the SFT trainer. It runs locally on one node or, with `[slurm]`, as a SLURM job on one or more nodes (see [Multi-node](#multi-node)).

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

The launcher writes the config and the tokenizer to `<run_dir>/configs/model` and trains from there, so the trainer, the HF export (`tools/convert_dcp_to_bf16.py`) and vLLM all see an ordinary model directory. The trainer logs the active non-embedding (routed experts counted at `top_k / num_experts`), non-embedding, embedding (input embedding and LM head) and total parameter counts at startup, and as `params/*` at step 0.

## Initialization

Weights start from a random init, unless the run resumes from a checkpoint.

| Field | Meaning |
|---|---|
| `model.init.std` | Every weight matrix and the embedding start from N(0, std) (default 0.02), scaled per role under CompleteP (below). Norm weights start at 1, biases at 0. |
| `model.init.seed` | Seed. Every rank initializes only its own shard, and the values depend only on the seed and the parameter name, not on the number of GPUs. |

## Parametrization

The optimizer gets one parameter group per role (embedding, hidden matrices, `o_proj`/`down_proj`, LM head, norms, biases; with MoE also experts and the router). By default (standard parametrization) every role uses `optim.lr`, `optim.eps` and `optim.weight_decay`, except norms and biases, which get no weight decay.

`[parametrization]` trains with [CompleteP](https://arxiv.org/abs/2505.01618) (muP extended to depth, alpha = 1), so that hyperparameters tuned on a small base shape carry over to wider and deeper models. `optim.lr`, `optim.eps`, `optim.weight_decay` and `model.init.std` are the base-shape values, and the base shape is set by `parametrization.base_hidden_size` and `parametrization.base_num_layers`. With `m_N = hidden_size / base_hidden_size` and `m_L = num_layers / base_num_layers`:

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
| MoE router | std / sqrt(m_N) | lr / m_N | wd m_N |

In MoE models, the expert matrices follow the hidden and `down_proj` rules per expert. The router maps the width to a fixed number of experts, so under Muon its learning rate falls as 1/m_N (with AdamW it follows the hidden rule).

Use QK-norm (`model.arch.use_qk_norm = true`) with CompleteP. Without it, CompleteP's attention collapses one or two LR steps above the optimum (the norm gains get no weight decay), and the cliff moves down with width. With QK-norm, CompleteP and the standard parametrization reach the same loss, and CompleteP's optimal LR stays fixed across widths (MoE QK-norm sweep). The per-role groups need AdamW or Muon and `model.offload = "none"`. The standard parametrization is CompleteP with the base shape equal to the model shape. Keep `head_dim` and the ratios `intermediate_size / hidden_size` (and `moe_intermediate_size / hidden_size`) and `num_attention_heads / num_key_value_heads` fixed across shapes.

## Learning-rate schedule

Pretraining uses the shared `[scheduler]`. Its default here is the `linear` scheduler with a warmup over 10% of the steps and a linear decay to `min_lr = 0` over the remaining 90%. Warmup and decay take a number of steps or a fraction of `max_steps` (rounded down), and `decay_shape` is `linear`, `cosine` or `1-sqrt`:

```toml
[scheduler]
type = "linear"
warmup_steps = 0.05
decay_steps = 0.2
decay_shape = "1-sqrt"
```

The schedule scales every param group's own learning rate, so ratios between groups hold. AdamW and Muon apply weight decay as `lr * weight_decay`, so the schedule also scales the decay. The scheduler state is checkpointed; resuming with `ckpt.skip_scheduler = true` rebuilds the schedule over the remaining steps.

## MoE load balancing

`[model.moe.balance]` balances the routed experts as in DeepSeek-V3, for every MoE family whose router has a selection bias (e.g. `glm4_moe`):

| Field | Meaning |
|---|---|
| `bias_update_rate` | Every step, each layer's selection bias moves by this step size (gamma) towards underloaded experts, using the step's global expert counts. The update is mean-centred. Default 3e-3; 0 turns it off. The bias starts at 0. |
| `bias_freeze` | Fraction of `max_steps` at the end during which the biases stop updating. Default 0.1. |
| `seq_aux_coeff` | Weight (alpha) of the sequence-wise balance loss, computed per packed row of `data.seq_len` tokens and added to the training loss. Default 1e-3; 0 turns it off. |

The metrics are `expert_load/*` (global balance of the step), `loss/seq_balance` (1 is balanced) and `router/bias/*`.

## Data

`data.type = "tokens"` reads [datatrove](https://github.com/huggingface/datatrove) `DocumentTokenizer` output: per source, a folder of `*.ds` token files with their `.ds.index` (document ends) and `.ds.metadata`. Each epoch reads every token of a source once: in blocks of 2^20 tokens in a random order and from a random start offset, both drawn from `data.seed`, the source and the epoch. Within an epoch, the rows of each group of 4,096 consecutive rows are read in a seeded random order, so the pieces of one long document spread over many steps. So every epoch has a new order and new row boundaries, and part of an epoch is a sample of the whole source, whatever the order of the files. Documents continue across rows, and attention restarts at every document, every row and every block boundary. Held-out sources are read in file-name order. Every document must end with the tokenizer's EOS; the trainer checks the first document of every source, which catches shards made with another tokenizer.

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

Batch sizes are in tokens, so changing `seq_len` keeps the tokens per step and the number of steps. Resume restores the data position on the same GPU layout and `data.num_workers`; resuming on a different layout is not supported yet. `data.type = "fake"` trains on random tokens, for smoke and throughput tests.

## Monitoring

Every step logs the loss, learning rate, gradient norms (total and per parameter group, `optim/grad_norm/<group>`), loss and grad-norm spike flags, per-source token counts and epochs (`progress/<source>/*`), throughput, MFU, data wait and peak memory (max and mean over ranks). MoE models add the expert balance of the step's global counts (`expert_load/*`, also per layer), `router/*` and `loss/seq_balance`. The step statistics are reduced once per step.

`[diagnostics]` (default: every 100 steps; `diagnostics = null` turns it off) adds the size of the optimizer update relative to the weights per parameter group (`update/rms_ratio/*`, `param/rms/*`) and a forward pass on a fixed probe batch: attention max logit and entropy per layer (`attn/*`, FlashAttention layers) and the residual-stream size per layer (`act/*`). It does not support context parallelism yet.

## Evals

`[val]` computes the held-out loss (nats per token), perplexity and bits per UTF-8 byte (comparable across tokenizers) of every source in `val.sources`, through the same loss code as training, so it also works with context parallelism. Each eval reads the same leading windows of every source: as many as fill whole micro-batches on every data-parallel rank. Evals run after the optimizer update, so the metrics at step N describe the same weights as the step-N checkpoint. With `eval_on_start` (default), they also run before the first step (logged as step 0), and they always run at the last step.

```toml
[val]
interval = 500

[val.sources]
web = { path = "data/web-val" }
```

**Eval suite (`[evals]`).** Every `interval` steps, at the last step and (with `eval_on_start`) on the weights the run starts from, the trainer scores the eval suite: log-likelihood sets (`ppl`, `ctxcont` and `pairs`, in `<data_dir>/<set>/{test.parquet,meta.json}`, the format of the pretrain-evals offline suite) with forward passes without gradients on the training model itself (same sharding, dtype and fusions). Requests are packed into document-masked rows that are dealt over the ranks; the LM head returns fp32 per-token log-probs for the duration of the evals. Every set is scored at each context length in `seq_lens` and logged as `eval/{set}/{metric}@{length}` (e.g. `eval/fwedu-holdout/bpb@4k`, plus accuracies and greedy exact match for multiple-choice and gold-answer sets); requests that are the same at several lengths (e.g. documents shorter than the smallest window) are scored once. Requires `model.cp = 1`.

```toml
[evals]
interval = 1000
data_dir = "/shared/evals-data"
sets = ["fwedu-holdout", "lambada", "math"]
seq_lens = [2048, 4096, 16384]
```

## Multi-node

`uv run pretrain` takes the same `[deployment]` and `[slurm]` blocks as `uv run sft` (see [Scaling § SLURM](scaling.md#slurm)) and submits the same bundled templates. The launcher writes the model dir on the submitting node, then submits one job in which every node runs `torchrun` on the pretrain trainer:

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

Row `w` of the token stream goes to data stream `w % num_streams`, with `num_streams = data_parallel_ranks * data.num_workers`, and step `s` (counted from 1) trains on rows `(s - 1) * rows_per_step` to `s * rows_per_step - 1`. So runs with different numbers of GPUs train on the same rows at every step, only split differently into micro-batches. The loss and gradient are normalized by the token count of the whole global batch, as on one GPU with gradient accumulation.
