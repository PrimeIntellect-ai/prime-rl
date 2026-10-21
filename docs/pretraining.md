# Pretraining

`uv run pretrain` trains a model from scratch on pre-tokenized token shards. It has its own config (`PretrainConfig`) and training loop, built from the same model setup, FSDP/EP/CP, optimizer, checkpointing and monitors as the SFT trainer.

## Table of Contents

- [Launch](#launch)
- [Model](#model)
- [Initialization](#initialization)
- [Optimizer Groups](#optimizer-groups)
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
| `model.init.std` | Every weight matrix and embedding (including the LM head, the router, expert and Gated Residual projections) starts from N(0, std) (default 0.02), with no width or depth scaling. Norm weights start at 1, biases at 0; zero-centred norm gains and the MoE selection biases start at 0. |
| `model.init.seed` | Seed. Every rank initializes only its own shard, and the values depend only on the seed and the parameter name, not on the number of GPUs. |

## Optimizer

Pretraining uses the optimizer setup of SFT and RL. With `optim.type = "muon"` (the default), every weight matrix except the embedding and the LM head is trained by Muon (`rms_norm` learning-rate adjustment; expert matrices per expert), and the embedding, the LM head, norms, biases and the Gated Residual projections (`hyper_connection`) by AdamW with `optim.eps`. `optim.type = "adamw"` trains everything with AdamW. `model.offload = "optimizer"` keeps the optimizer states in CPU RAM between steps; `"full"` is not supported.

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
| `bias_update` | `quantile` (default): quantile balancing. Every step, each layer's bias is set so that every expert would have been selected by `num_experts_per_tok / num_experts` of the step's tokens (null experts included), estimated as `quantile_estimator` sets; no rate. Best with `router_bias_on_logits` (on by default). `sign`: the update below. The bias starts at 0. |
| `quantile_estimator` | How `quantile` estimates the thresholds. `histogram` (default; Kimi K3 report, §2.3.3): one histogram of the margins per expert over the whole step, summed over the ranks with one all-reduce, read with in-bin interpolation; the quantile of the pooled step, independent of how the tokens are split. `micro_batch_mean`: each micro-batch's exact quantile, averaged over the micro-batches and ranks. |
| `quantile_bins` | Bins of the `histogram` estimator (default 1024). Its grid is the previous step's global margin range, `[-1, 1]` on the first step after a start or resume. |
| `bias_update_rate` | With `sign`: every step, each layer's selection bias moves by this step size (gamma) towards underloaded experts, using the step's global expert counts. The update is mean-centred. Default 3e-3; 0 turns it off. |
| `bias_freeze` | Fraction of `max_steps` at the end during which the biases stop updating. Default 0.1. |
| `seq_aux_coeff` | Weight (alpha) of the sequence-wise balance loss, computed per packed row of `data.seq_len` tokens and added to the training loss. Default 1e-3; 0 turns it off. |

The metrics are `expert_load/*` (global balance of the step), `loss/seq_balance` (1 is balanced) and `router/bias/*`. Every 32 steps the trainer also records each layer's per-expert selection bias, router-row norm and load in `moe_trace.npz` in the run directory.

## Data

`data.type = "tokens"` reads [datatrove](https://github.com/huggingface/datatrove) `DocumentTokenizer` output: folders of `*.ds` token files with their `.ds.index` (document ends) and `.ds.metadata`. The files of all folders in `data.paths` (in folder order, then file-name order) are read as one token pool. Each epoch reads every row of the pool once, in a random order over the whole pool and from a random start offset, both drawn from `data.seed` and the epoch (the last `seq_len` tokens or fewer of an epoch are skipped). So every epoch has a new order and new row boundaries, and part of an epoch is a sample of the whole pool, whatever the order of the files. Every row is a random read (about 33 ms each from a cold cache on CephFS); at the screening shapes, `data.num_workers = 2` with the default prefetch hides this (`time/data` stays at 0). Check `time/data` on other storage. Documents continue across rows, and attention and positions restart at every document and every row; with `data.document_masking = false`, a row is one sequence (attention spans the whole row and positions run over it, as in GPT-style training without document masking). Every document must end with the tokenizer's EOS; the trainer checks the first document of every folder, which catches shards made with another tokenizer.

Every folder contributes rows in proportion to its tokens, so mixtures are made offline, as folders whose token counts follow the mixture weights. At startup the trainer computes how many passes over the pool the run makes and fails above `data.max_epochs`.

```toml
[data]
type = "tokens"
seq_len = 8192        # tokens per row
batch_size = 32       # rows per optimizer step, over all GPUs
micro_batch_size = 8  # rows per forward pass on one GPU
paths = ["data/web", "data/code"]
```

Batch sizes are in rows, as in SFT: a step trains on `batch_size * seq_len` tokens, so changing `seq_len` changes the tokens per step. Resume restores the data position on the same GPU layout and `data.num_workers`; resuming on a different layout is not supported yet. `data.type = "fake"` trains on random tokens, for smoke and throughput tests.

## Monitoring

Every step logs the loss, learning rate, gradient norms (total and per parameter group, `optim/grad_norm/<group>`), loss and grad-norm spike flags, the rows, tokens and epochs trained on (`progress/num_samples`, `progress/num_tokens`, `progress/epochs`), throughput, MFU, data wait and peak memory (max and mean over ranks). MoE models add the expert balance of the step's global counts (`expert_load/*`, also per layer), `router/*` and `loss/seq_balance`. The step statistics are reduced once per step.

## Evals

`[evals]`: every `interval` steps, at the last step and (with `eval_on_start`, default) on the weights the run starts from, the trainer scores its eval sets with forward passes without gradients on the training model itself (same sharding, dtype and fusions). Evals run after the optimizer update, so the metrics at step N describe the same weights as the step-N checkpoint. A set is either

- the name of a log-likelihood set of the pretrain-evals offline suite (`ppl`, `ctxcont` or `pairs`, in `<data_dir>/<set>/{test.parquet,meta.json}`), or
- a held-out folder of datatrove token shards (`{ name = ..., path = ... }`), scored like a `ppl` set: every document of the folder as stored, in windows of each of `seq_lens`, every token after its first scored (loss in nats per token and bits per UTF-8 byte, comparable across tokenizers). Each eval scores the whole folder at every length, so keep held-out folders small: at d512 on one H200, a token set costs about 1 s per 200K tokens per length (5M tokens at `seq_lens = [2048, 8192]`: 40 s, against 280 s for the 58-set suite). Because every document is scored from its start, these losses are lower than those of the former `[val]` (by 0.014-0.027 nats at d512), which started rows mid-document and predicted each document's first token across the document boundary: don't compare old `val/*` numbers with `eval/*` token-set numbers.

Requests are packed into document-masked rows of `evals.micro_batch_tokens` tokens (default: the training micro-batch, `data.micro_batch_size * data.seq_len`, or the longest of `seq_lens`, whichever is larger) that are dealt over the ranks; the LM head returns fp32 per-token log-probs for the duration of the evals. Every set is scored at each context length in `seq_lens` (default: `data.seq_len`) and logged as `eval/{set}/{metric}@{length}` (e.g. `eval/fwedu-holdout/bpb@4k`, plus accuracies and greedy exact match for multiple-choice and gold-answer sets); requests that are the same at several lengths (e.g. documents shorter than the smallest window) are scored once. Requests of the pretrain-evals sets start with BOS only if the training documents do (read off the first document of every training folder at startup), so the evals see the training format; without BOS, the first token of a `ppl` document or `pairs` sentence is only context (`ppl` bpb leaves out its bytes). Requires `model.cp = 1`. Evals run without gradients, so longer rows can fit: in our tests, eval memory stayed at most 0.6x the training peak, and at d512 (8k training micro-batches), rows of 32k instead of 16k tokens cut the eval time by 19%.

```toml
[evals]
interval = 1000
data_dir = "/shared/evals-data"
sets = ["fwedu-holdout", "lambada", "math", { name = "heldout", path = "data/dolma3-50b-dsv4/mix-val-5M" }]
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
