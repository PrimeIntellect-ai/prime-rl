# Pretraining

`uv run pretrain` trains a model from scratch on pre-tokenized token shards. It has its own config (`PretrainConfig`) and training loop, built from the same model setup, FSDP/EP/CP, optimizer, checkpointing and monitors as the SFT trainer.

## Table of Contents

- [Launch](#launch)
- [Model](#model)
- [Initialization](#initialization)
- [Parametrization](#parametrization)
- [Learning-Rate Schedule](#learning-rate-schedule)
- [MoE Load Balancing](#moe-load-balancing)
- [Data](#data)
- [Monitoring](#monitoring)
- [Evals](#evals)
- [Multi-Node](#multi-node)

## Launch

```bash
uv run pretrain @ examples/pretrain/prime_moe.toml        # MoE example on token shards
uv run pretrain @ configs/debug/fake/pretrain.toml        # 5-step smoke test on random tokens
```

It runs locally on one node or, with `[slurm]`, as a SLURM job on one or more nodes (see [Multi-Node](#multi-node)).

## Model

The model is set up as in SFT and RL: `model.name` is an HF model id or a local model directory with a `config.json`. Its weights are not loaded. The model starts from a random init (below) or from a checkpoint. A new architecture is a directory with only a `config.json` for a model type in `prime_rl.trainer.models`; [`examples/pretrain/moe-d512`](../examples/pretrain/moe-d512/config.json) is the one in the example. `vocab_size` and the BOS, EOS and pad ids in the config must fit the tokenizer. The checked-in configs round `len(tokenizer)` up to a multiple of 128 and use the tokenizer's special ids, with EOS as the pad id if the tokenizer has none.

```toml
[model]
name = "examples/pretrain/moe-d512"

[tokenizer]
name = "tokenizers/deepseek-v4-flash-base"  # defaults to model.name
```

MoE models use `prime_moe`, pretraining's own model (`prime_rl.trainer.models.prime_moe`), so architecture work does not touch the shared MoE code that SFT and RL use. Its base is a decoder of grouped-query attention (optional QK norm, partial RoPE) and MoE layers with sigmoid-scored token-choice routed experts, a selection bias and shared experts, after leading dense layers. Its module names and state-dict layout match `glm4_moe`'s, so checkpoints convert the same way. On top of the base:
- Gated Residual connections (Qwen3.8-Next): `hc_count` residual streams (above 1) with read gates of rank `hc_lowrank` replace the pre-norms and the final norm. Their projections train with AdamW.
- Null experts (AdaMoE): `n_null_experts` zero-output experts the router can select next to the routed ones, so tokens use between 0 and `num_experts_per_tok` routed experts. Only the (token, routed expert) pairs are dispatched. The metrics are `router/k/*`.
- Exclusive self attention (XSA): `exclusive_attention` removes from each head's output its component along the token's own value vector.
- Gated attention (Qwen3-Next): `attn_output_gate` multiplies the attention output elementwise by `sigmoid(g_proj(x))`, computed from the block input, before `o_proj` (after XSA).
- LatentMoE: with `latent_dim`, the routed experts work in that width (projection down and RMS norm before the dispatch, projection up after the combine). The router and the shared experts stay full width. With `latent_dim = d/2`, doubling the routed experts, the null experts and `num_experts_per_tok` (and `routed_scaling_factor`, which keeps each expert's gate) keeps the expert parameters and routed FLOPs per token (Kimi K3, NVIDIA LatentMoE). `latent_norm` puts its RMS norm on the expert input (`"input"`, Marin) or on the combined expert output before the up-projection (`"output"`, Kimi K3's Stable LatentMoE).
- `expert_activation = "situ_glu"`: Kimi K3's SiTU-GLU in the routed and shared experts, SwiGLU with both factors soft-capped (`4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)`).
- `router_bias_on_logits` (default true) adds the selection bias to the router logits instead of the sigmoid scores (for quantile balancing).

The trainer and the HF export (`tools/convert_dcp_to_bf16.py`) read the same model directory and tokenizer; vLLM does not support `prime_moe` yet. The trainer logs the active non-embedding (routed experts counted at `top_k / num_experts`), non-embedding, embedding (input embedding and LM head) and total parameter counts at startup, and as `params/*` at step 0.

## Initialization

Weights start from a random init, unless the run resumes from a checkpoint. The init lives with the models (`prime_rl.trainer.models.init`) and runs in the shared `setup_model` when it gets an `init` config. SFT and RL do not pass one, so their behavior is unchanged.

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

Use QK-norm (`"use_qk_norm": true` in the model config) with CompleteP. Without it, CompleteP's attention collapses one or two LR steps above the optimum (the norm gains get no weight decay), and the cliff moves down with width. With QK-norm, CompleteP and the standard parametrization reach the same loss, and CompleteP's optimal LR stays fixed across widths (MoE QK-norm sweep). The per-role groups need AdamW or Muon and `model.offload = "none"`. The standard parametrization is CompleteP with the base shape equal to the model shape. Keep `head_dim` and the ratios `intermediate_size / hidden_size` (and `moe_intermediate_size / hidden_size`) and `num_attention_heads / num_key_value_heads` fixed across shapes.

## Learning-Rate Schedule

Pretraining uses the shared `[scheduler]`. Its default here is the `linear` scheduler with a warmup over 10% of the steps and a linear decay to `min_lr = 0` over the remaining 90%. Warmup and decay take a number of steps or a fraction of `max_steps` (rounded down), and `decay_shape` is `linear`, `cosine` or `1-sqrt`:

```toml
[scheduler]
type = "linear"
warmup_steps = 0.05
decay_steps = 0.2
decay_shape = "1-sqrt"
```

The schedule scales every param group's own learning rate, so ratios between groups hold. AdamW and Muon apply weight decay as `lr * weight_decay`, so the schedule also scales the decay. The scheduler state is checkpointed; resuming with `ckpt.skip_scheduler = true` rebuilds the schedule over the remaining steps.

## MoE Load Balancing

`[model.moe.balance]` balances the routed experts of `prime_moe` as in DeepSeek-V3:

| Field | Meaning |
|---|---|
| `bias_update` | `quantile` (default): quantile balancing. Every step, each layer's bias is set so that every expert would have been selected by `num_experts_per_tok / num_experts` of the step's tokens (null experts included), estimated per micro-batch and averaged over the micro-batches and ranks; no rate. Best with `router_bias_on_logits` (on by default). `sign`: the update below. The bias starts at 0. |
| `bias_update_rate` | With `sign`: every step, each layer's selection bias moves by this step size (gamma) towards underloaded experts, using the step's global expert counts. The update is mean-centred. Default 3e-3; 0 turns it off. |
| `bias_freeze` | Fraction of `max_steps` at the end during which the biases stop updating. Default 0.1. |
| `seq_aux_coeff` | Weight (alpha) of the sequence-wise balance loss, computed per packed row of `data.seq_len` tokens and added to the training loss. Default 1e-3; 0 turns it off. |

The metrics are `expert_load/*` (global balance of the step), `loss/seq_balance` (1 is balanced) and `router/bias/*`. Every 32 steps the trainer also records each layer's per-expert selection bias, router-row norm and load in `moe_trace.npz` in the run directory.

## Data

`data.type = "tokens"` reads [datatrove](https://github.com/huggingface/datatrove) `DocumentTokenizer` output: per source, a folder of `*.ds` token files with their `.ds.index` (document ends) and `.ds.metadata`. The files of a source are read as one stream. Each epoch reads every row of a source once, in a random order over the whole source and from a random start offset, both drawn from `data.seed`, the source and the epoch (the last `seq_len` tokens or fewer of an epoch are skipped). So every epoch has a new order and new row boundaries, and part of an epoch is a sample of the whole source, whatever the order of the files. Every row is a random read (about 33 ms each from a cold cache on CephFS); at the screening shapes, `data.num_workers = 2` with the default prefetch hides this (`time/data` stays at 0). Check `time/data` on other storage. Documents continue across rows, and attention restarts at every document and every row. Held-out sources are read in file-name order. Every document must end with the tokenizer's EOS; the trainer checks the first document of every source, which catches shards made with another tokenizer.

Each row comes from one source, drawn with the mixture `data.weights`. At startup the trainer computes how many passes over each source the run makes and fails above `data.max_epochs`.

```toml
[data]
type = "tokens"
seq_len = 4096
batch_tokens = 262144       # per optimizer step, over all GPUs
micro_batch_tokens = 65536  # per forward pass on one GPU
max_epochs = 3
weights = { web = 0.8, code = 0.2 }

[data.sources]
web = { path = "data/web" }
code = { path = "data/code" }
```

Batch sizes are in tokens, so changing `seq_len` keeps the tokens per step and the number of steps. Resume restores the data position on the same GPU layout and `data.num_workers`; resuming on a different layout is not supported yet. `data.type = "fake"` trains on random tokens, for smoke and throughput tests.

## Monitoring

Every step logs the loss, learning rate, gradient norms (total and per parameter group, `optim/grad_norm/<group>`), loss and grad-norm spike flags, per-source token counts and epochs (`progress/<source>/*`), throughput, MFU, data wait and peak memory (max and mean over ranks). MoE models add the expert balance of the step's global counts (`expert_load/*`, also per layer), `router/*` and `loss/seq_balance`. The step statistics are reduced once per step.

## Evals

`[val]` computes the held-out loss (nats per token), perplexity and bits per UTF-8 byte (comparable across tokenizers) of every source in `val.sources`, through the same loss code as training, so it also works with context parallelism. Each eval reads the same leading windows of every source: as many as fill whole micro-batches on every data-parallel rank. Evals run after the optimizer update, so the metrics at step N describe the same weights as the step-N checkpoint. With `eval_on_start` (default), they also run before the first step (logged as step 0), and they always run at the last step.

```toml
[val]
interval = 500

[val.sources]
web = { path = "data/web-val" }
```

**Eval suite (`[evals]`).** Every `interval` steps, at the last step and (with `eval_on_start`) on the weights the run starts from, the trainer scores the eval suite: log-likelihood sets (`ppl`, `ctxcont` and `pairs`, in `<data_dir>/<set>/{test.parquet,meta.json}`, the format of the pretrain-evals offline suite) with forward passes without gradients on the training model itself (same sharding, dtype and fusions). Requests are packed into document-masked rows of `evals.micro_batch_tokens` tokens (default: `data.micro_batch_tokens` or the longest of `seq_lens`, whichever is larger) that are dealt over the ranks; the LM head returns fp32 per-token log-probs for the duration of the evals. Every set is scored at each context length in `seq_lens` and logged as `eval/{set}/{metric}@{length}` (e.g. `eval/fwedu-holdout/bpb@4k`, plus accuracies and greedy exact match for multiple-choice and gold-answer sets); requests that are the same at several lengths (e.g. documents shorter than the smallest window) are scored once. Requests start with BOS only if the training documents do (read off the first document of every training source at startup), so the evals see the training format; without BOS, the first token of a `ppl` document or `pairs` sentence is only context (`ppl` bpb leaves out its bytes). Requires `model.cp = 1`. Evals run without gradients, so longer rows can fit: in our tests, eval memory stayed at most 0.6x the training peak, and at d512 (8k training micro-batches), rows of 32k instead of 16k tokens cut the eval time by 19%.

```toml
[evals]
interval = 1000
data_dir = "/shared/evals-data"
sets = ["fwedu-holdout", "lambada", "math"]
seq_lens = [2048, 4096, 16384]
```

## Multi-Node

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
