# Reference

This page documents every field accepted by every prime-rl entrypoint. It is
auto-generated; do not edit by hand.

## Table of Contents

- [`rl` — Full RL training](#rl)
  - [`trainer`](#rl-trainer)
  - [`orchestrator`](#rl-orchestrator)
  - [`inference`](#rl-inference)
  - [`log`](#rl-log)
  - [`ckpt`](#rl-ckpt)
  - [`wandb`](#rl-wandb)
  - [`model`](#rl-model)
  - [`tokenizer`](#rl-tokenizer)
  - [`weight_broadcast`](#rl-weight-broadcast)
  - [`slurm`](#rl-slurm)
  - [`experimental`](#rl-experimental)
  - [`deployment`](#rl-deployment)
- [`sft` — Supervised fine-tuning](#sft)
  - [`model`](#sft-model)
  - [`tokenizer`](#sft-tokenizer)
  - [`renderer`](#sft-renderer)
  - [`val`](#sft-val)
  - [`ckpt`](#sft-ckpt)
  - [`log`](#sft-log)
  - [`wandb`](#sft-wandb)
  - [`bench`](#sft-bench)
  - [`gc`](#sft-gc)
  - [`heartbeat`](#sft-heartbeat)
  - [`slurm`](#sft-slurm)
  - [`experimental`](#sft-experimental)
  - [`data`](#sft-data)
  - [`optim`](#sft-optim)
  - [`scheduler`](#sft-scheduler)
  - [`deployment`](#sft-deployment)
- [`trainer` — Standalone trainer](#trainer)
  - [`model`](#trainer-model)
  - [`tokenizer`](#trainer-tokenizer)
  - [`data`](#trainer-data)
  - [`ckpt`](#trainer-ckpt)
  - [`log`](#trainer-log)
  - [`wandb`](#trainer-wandb)
  - [`bench`](#trainer-bench)
  - [`gc`](#trainer-gc)
  - [`heartbeat`](#trainer-heartbeat)
  - [`metrics_server`](#trainer-metrics-server)
  - [`experimental`](#trainer-experimental)
  - [`loss`](#trainer-loss)
  - [`optim`](#trainer-optim)
  - [`scheduler`](#trainer-scheduler)
  - [`weight_broadcast`](#trainer-weight-broadcast)
  - [`rollout_transport`](#trainer-rollout-transport)
- [`orchestrator` — Standalone orchestrator](#orchestrator)
  - [`student`](#orchestrator-student)
  - [`teacher`](#orchestrator-teacher)
  - [`train`](#orchestrator-train)
  - [`tokenizer`](#orchestrator-tokenizer)
  - [`renderer`](#orchestrator-renderer)
  - [`optim`](#orchestrator-optim)
  - [`eval`](#orchestrator-eval)
  - [`buffer`](#orchestrator-buffer)
  - [`log`](#orchestrator-log)
  - [`wandb`](#orchestrator-wandb)
  - [`prime_monitor`](#orchestrator-prime-monitor)
  - [`ckpt`](#orchestrator-ckpt)
  - [`heartbeat`](#orchestrator-heartbeat)
  - [`experimental`](#orchestrator-experimental)
  - [`filters.<n>` (list item)](#orchestrator-filters)
  - [`weight_broadcast`](#orchestrator-weight-broadcast)
  - [`rollout_transport`](#orchestrator-rollout-transport)
- [`inference` — Standalone vLLM server](#inference)
  - [`server`](#inference-server)
  - [`model`](#inference-model)
  - [`parallel`](#inference-parallel)
  - [`weight_broadcast`](#inference-weight-broadcast)
  - [`kv_cache_offload`](#inference-kv-cache-offload)
  - [`slurm`](#inference-slurm)
  - [`experimental`](#inference-experimental)
  - [`deployment`](#inference-deployment)

---

<a id="rl"></a>
## `rl` — Full RL training

The `rl` entrypoint composes a trainer, orchestrator, and (optionally) inference server into a single co-located deployment. Sub-configs under `[trainer]`, `[orchestrator]`, and `[inference]` mirror the standalone entrypoints below, with shared knobs (model name, output dir, W&B run name, …) lifted to the top level so they only need to be set once.

_Defined in_ `prime_rl.configs.rl.RLConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `Path` | `'outputs'` | Output directory. Should be unique per experiment. |
| `clean_output_dir` | `bool` | `False` | Delete the output directory before starting training. Required to overwrite an output directory that contains checkpoints from a previous run when not resuming. |
| `max_steps` | `int | None` | `None` | Shared maximum training steps. If None, falls back to the sub-config ``max_steps``. |
| `seq_len` | `int | None` | `None` | Shared sequence length. Propagates to ``trainer.model.seq_len`` and ``orchestrator.seq_len`` only when those values were not explicitly set; explicit per-component values always win. |
| `max_async_level` | `int | None` | `None` | Shared async level. If None, falls back to the sub-config ``max_async_level``. |
| `bench` | `bool` | `False` | Benchmark mode. Sets trainer and orchestrator to benchmark mode and, when set, suffixes the W&B project with ``-bench``. |
| `dry_run` | `bool` | `False` | Only validate and dump resolved configs, then exit early. |

<a id="rl-trainer"></a>
### `trainer`

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.output_dir` | `Path` | `'outputs'` | Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node. |
| `trainer.matmul_precision` | `'highest' | 'high' | 'medium'` | `'high'` | Precision for float32 matrix multiplications. ``highest`` is full FP32 (required on ROCm/AMD GPUs to avoid catastrophic precision loss in softmax over large vocabularies). ``high`` enables TF32 on NVIDIA GPUs for a speedup with minor precision tradeoff. See ``torch.set_float32_matmul_precision``. |
| `trainer.max_steps` | `int | None` | `None` | Maximum number of training steps. If None, runs indefinitely. |
| `trainer.max_async_level` | `int` | `1` | _≥0._ Maximum steps inference can be ahead of training (how off-policy inference can be). Higher values yield better throughput via async execution at the cost of policy lag; ``0`` is fully synchronous. |
| `trainer.enable_router_replay` | `bool` | `False` | Return routed experts in the batch so the trainer can replay routing. Requires ``enable_return_routed_experts=true`` on the vLLM server (or ``--enable-return-routed-experts``) and is only supported for custom models. |
| `trainer.memory_profiler_path` | `Path | None` | `None` | Path to write the memory profile to. |
| `trainer.trace_path` | `Path | None` | `None` | Path to write the PyTorch profiler trace to. |
| `trainer.dist_timeout_seconds` | `int` | `600` | Timeout in seconds for torch distributed ops. |
| `trainer.max_concurrent_runs` | `int` | `1` | _≥1._ Maximum number of concurrent runs to allow. If 1, only one run may run at a time. |

<a id="rl-trainer-model"></a>
#### `trainer.model`

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `trainer.model.trust_remote_code` | `bool` | `False` | Trust remote code when initializing the tokenizer. |
| `trainer.model.seq_len` | `int` | `2048` | Sequence length the model is trained on. |
| `trainer.model.attn` | `'eager' | 'sdpa' | 'flash_attention_2' | 'flash_attention_3' | 'fa4'` | `'flash_attention_2'` | Attention implementation. With CP enabled, ring attention uses the matching kernel family (FA2/FA3/FA4). |
| `trainer.model.fsdp_cpu_offload` | `bool` | `False` | Enable FSDP CPU offloading for parameters, gradients, and optimizer states. Uses pinned memory for efficient CPU↔GPU transfers. |
| `trainer.model.optim_cpu_offload` | `bool` | `False` | Offload only optimizer states (momentum, variance) to CPU, keeping weights on GPU. Avoids the H2D all-gather overhead of FSDP CPU offload while still saving GPU memory. |
| `trainer.model.reshard_after_forward` | `bool` | `True` | Reshard the model after each forward pass. |
| `trainer.model.dp_replicate` | `int` | `1` | Data parallel dim where model weights are replicated. |
| `trainer.model.ep` | `int` | `1` | Expert parallelism degree for MoE layers. 1 disables EP. |
| `trainer.model.ep_comm_backend` | `'torch' | 'deepep'` | `'torch'` | Communication backend for expert parallelism. ``torch`` uses TorchTitan all-to-all collectives; ``deepep`` uses DeepEP custom kernels. |
| `trainer.model.deepep_num_sms` | `int` | `20` | _≥1._ SMs allocated for DeepEP intranode dispatch/combine kernels. Also determines internode RDMA channel count (``num_channels = num_sms / 2``). Lower values leave more SMs for compute; higher values speed up dispatch/combine. The optimal value depends on EP degree and hardware. Only used when ``ep_comm_backend='deepep'``. |
| `trainer.model.deepep_token_chunk_size` | `int | None` | `None` | _≥1._ Token chunk size for DeepEP MoE pipelining. When set, DeepEP dispatch for chunk i+1 is launched while experts compute chunk i. Only used when ``ep_comm_backend='deepep'``. |
| `trainer.model.cp` | `int` | `1` | Context parallelism degree. 1 disables CP. |
| `trainer.model.cp_style` | `'ring' | 'ulysses'` | `'ring'` | CP communication style. ``ring`` uses ring-attention all-gather/reduce-scatter (requires custom kernels per attention type). ``ulysses`` uses all-to-all to redistribute Q/K/V from sequence-sharded to head-sharded, runs vanilla attention locally on the full sequence, then all-to-all back — works out-of-the-box with any attention kernel (softmax FA, linear attention, mamba, etc.). |
| `trainer.model.impl` | `'hf' | 'custom' | 'auto'` | `'auto'` | Model implementation. ``auto`` selects ``custom`` if supported by the model, otherwise ``hf``. |
| `trainer.model.optimization_dtype` | `'bfloat16' | 'float32'` | `'float32'` | dtype for model optimization. |
| `trainer.model.reduce_dtype` | `'bfloat16' | 'float32'` | `'float32'` | dtype for gradient/parameter reductions. |
| `trainer.model.moe_use_grouped_mm` | `bool` | `True` | Use grouped mm for MoE layers. Requires compute capability ≥ 9.0. |
| `trainer.model.fp8` | `bool` | `False` | FP8 training via DeepGEMM. Replaces ``nn.Linear`` with FP8 blockwise linear and uses FP8 grouped GEMM for MoE experts. Requires SM90 (Hopper) GPUs and ``model.impl='custom'``. |
| `trainer.model.freeze_moe_router` | `bool` | `False` | Freeze MoE router parameters during training. |
| `trainer.model.fused_lm_head_token_chunk_size` | `int | 'auto' | 'disabled'` | `'disabled'` | Flattened token chunk size for the fused LM head. ``int >= 1`` sets the tokens per LM-head chunk explicitly; ``auto`` auto-enables (RL training picks 8192); ``disabled`` uses the vanilla LM head. Integer values aren't supported for SFT training. |

<a id="rl-trainer-model-vlm"></a>
##### `trainer.model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `trainer.model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `trainer.model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="rl-trainer-model-compile"></a>
##### `trainer.model.compile`

Compile the model with ``torch.compile``.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.model.compile.fullgraph` | `bool` | `False` | Compile transformer blocks with ``fullgraph=True``. |

<a id="rl-trainer-model-ac"></a>
##### `trainer.model.ac`

Activation checkpointing configuration. If None, activation checkpointing is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.model.ac.mode` | `'full' | 'selective'` | `'full'` | ``full`` checkpoints whole transformer blocks; ``selective`` checkpoints only the subcomponents listed in ``targets`` inside supported custom decoder layers. |
| `trainer.model.ac.freq` | `int` | `1` | _≥1._ Apply activation checkpointing to every N layers. |
| `trainer.model.ac.targets` | `list[str]` | `['norm']` | Selective checkpoint targets. ``norm`` checkpoints every norm module inside selected layers. ``attn_proj`` checkpoints projection-side attention work outside the kernel (input/output projections, attention-local norms, RoPE, gating, model-specific MLA projection helpers). ``mlp`` checkpoints the entire dense MLP forward (not for MoE). ``mla_up_proj`` checkpoints MLA Q/KV up-projection where supported. ``routed_experts`` checkpoints routed expert compute in MoE layers (including LatentMoE). ``linear_attn`` checkpoints non-softmax token mixers (NemotronH Mamba, Qwen3.5-MoE GatedDeltaNet, AFMoE sliding-window attention). |

<a id="rl-trainer-model-ac-offloading"></a>
##### `trainer.model.ac_offloading`

Activation offloading configuration. If None, activation offloading is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.model.ac_offloading.pin_memory` | `bool` | `True` | Pin offloaded activations to CPU memory. |
| `trainer.model.ac_offloading.max_inflight_activations` | `int` | `5` | _≥1._ Max activations kept in flight while offloading. More activations smooth overlap at the cost of GPU memory. |

<a id="rl-trainer-model-index-cache"></a>
##### `trainer.model.index_cache`

DSA IndexCache sub-configuration. If set, sparse-attention top-k indices are reused across decoder layers per the configured schedule (mirrors vLLM's IndexCache HF overrides). If None, every layer recomputes its own indices.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.model.index_cache.topk_freq` | `int` | `1` | _≥1._ Recompute DSA top-k indices every N layers; intervening layers reuse the cached indices. ``1`` recomputes every layer (effectively no reuse). Mirrors vLLM's ``index_topk_freq`` HF override. |
| `trainer.model.index_cache.topk_pattern` | `str | None` | `None` | Optional per-layer schedule that overrides ``topk_freq``. ``'F'`` computes fresh indices for that layer; ``'S'`` reuses the previously cached indices. Length should match the number of decoder layers. |

<a id="rl-trainer-model-lora"></a>
##### `trainer.model.lora`

LoRA configuration. If None, LoRA is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.model.lora.rank` | `int` | `16` | _≥1._ Rank of the low-rank decomposition matrices. |
| `trainer.model.lora.alpha` | `float` | `32.0` | _≥0._ LoRA scaling parameter. |
| `trainer.model.lora.dropout` | `float` | `0.0` | _≥0, ≤1._ LoRA dropout rate. |
| `trainer.model.lora.target_modules` | `list[str]` | `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'experts', 'fc1_latent_proj', 'fc2_latent_proj']` | Module names or regex patterns to apply LoRA to. Simple names (e.g. ``q_proj``) match any component in the module path; regex patterns match anywhere in the name. Names unknown to the current model are silently ignored, so defaults cover multiple architectures. NemotronH note: ``experts`` matches NonGatedGroupedExperts inside LatentMoE; ``fc1_latent_proj``/``fc2_latent_proj`` adapt the latent up/down projections. Add ``in_proj``/``out_proj`` to also LoRA Mamba. |
| `trainer.model.lora.modules_to_save` | `list[str]` | `[]` | Module names or regex patterns to keep fully trainable (not freeze). Same matching rules as ``target_modules``. |

<a id="rl-trainer-model-debug"></a>
##### `trainer.model.debug`

Debugging knobs for the model and distributed training.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.model.debug.num_layers` | `int | None` | `None` | Override the number of transformer layers (truncates the model). |
| `trainer.model.debug.random_init` | `bool` | `False` | Randomly initialize the model instead of loading weights. |
| `trainer.model.debug.force_balanced_routing` | `bool` | `False` | Replace MoE token-choice routing with a round-robin assignment so every expert sees an equal share. Intended for fake-data smoke tests where untrained routing would otherwise OOM under severe imbalance. Gating scores are still gathered from the override indices so the forward pass stays consistent. |

<a id="rl-trainer-tokenizer"></a>
#### `trainer.tokenizer`

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.tokenizer.name` | `str | None` | `None` | Tokenizer name or path. If None, the model's default tokenizer is used. |
| `trainer.tokenizer.trust_remote_code` | `bool | None` | `None` | Trust remote code when initializing the tokenizer. If None, inherits the model's ``trust_remote_code`` setting. |
| `trainer.tokenizer.chat_template` | `str | None` | `None` | Chat template for the tokenizer. Either a Jinja2 template string or a path to a template file. If None, the tokenizer's default chat template is used. |

<a id="rl-trainer-data"></a>
#### `trainer.data`

<a id="rl-trainer-data-fake"></a>
##### `trainer.data.fake`

Use a fake data loader sampling random micro-batches (for debugging).

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.data.fake.batch_size` | `int` | `2` | _≥1._ Batch size of the fake data loader. |
| `trainer.data.fake.generate_samples` | `bool` | `False` | Generate separate samples and pack them into a single micro-batch instead of using random tensors. |

<a id="rl-trainer-ckpt"></a>
#### `trainer.ckpt`

Full training-state checkpoint configuration (model + optimizer + scheduler). If None, no resume-capable checkpoints are written.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.ckpt.output_dir` | `Path | None` | `None` | Override directory for checkpoints and weights. If set, checkpoints and weight snapshots are written here instead of under the trainer ``output_dir`` — useful for writing large checkpoints to a separate storage volume. |
| `trainer.ckpt.interval` | `int | None` | `None` | _≥1._ Interval at which to save the training checkpoint. If None, only checkpoints at the end of training. |
| `trainer.ckpt.skip_gather_master_weights` | `bool` | `False` | Skip gathering and saving HF-compatible weight checkpoints. Useful for large models where the gather is expensive and only DCP checkpoints are needed. |
| `trainer.ckpt.weights_only` | `bool` | `False` | Save only weight checkpoints (no optimizer/scheduler state). Much faster and smaller than full checkpoints, but cannot resume training. |
| `trainer.ckpt.resume_step` | `int | None` | `None` | _≥-1._ Step to resume training from. None starts from scratch; ``-1`` restarts from the latest checkpoint available. |
| `trainer.ckpt.keep_last` | `int | None` | `None` | _≥1._ Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency. |
| `trainer.ckpt.keep_interval` | `int | None` | `None` | _≥1._ Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping. |
| `trainer.ckpt.skip_progress` | `bool` | `False` | Skip loading the progress from checkpoint. |
| `trainer.ckpt.skip_scheduler` | `bool` | `False` | Skip loading the scheduler from checkpoint. |
| `trainer.ckpt.skip_dataloader` | `bool` | `False` | Skip loading the dataloader from checkpoint. |
| `trainer.ckpt.skip_optimizer` | `bool` | `False` | Skip loading the optimizer state from checkpoint. |

<a id="rl-trainer-ckpt-weights"></a>
##### `trainer.ckpt.weights`

Weight-checkpoint sub-configuration. If None, no HF-compatible weight checkpoints are written.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.ckpt.weights.save_sharded` | `bool` | `True` | Save the weight checkpoint in sharded format. |
| `trainer.ckpt.weights.save_format` | `'safetensors' | 'torch'` | `'safetensors'` | Weight checkpoint serialization format. |
| `trainer.ckpt.weights.save_adapter_separately` | `bool` | `False` | Save LoRA adapters separately before merging into full model weights. |

<a id="rl-trainer-log"></a>
#### `trainer.log`

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.log.level` | `str` | `'info'` | Log level for the process. Defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``. |
| `trainer.log.vf_level` | `str` | `'info'` | Log level for the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) package. Defaults to ``$PRIME_VF_LOG_LEVEL`` if set, else ``info``. |
| `trainer.log.json_logging` | `bool` | `False` | Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.). |
| `trainer.log.log_data` | `bool` | `False` | Log the first data sample at startup. |
| `trainer.log.ranks_filter` | `list[int]` | `[0]` | Trainer ranks to show in console output. Passed to ``torchrun --local-ranks-filter``. |

<a id="rl-trainer-wandb"></a>
#### `trainer.wandb`

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.wandb.project` | `str` | `'prime-rl'` | W&B project to log to. |
| `trainer.wandb.entity` | `str | None` | `None` | W&B entity to log to. |
| `trainer.wandb.name` | `str | None` | `None` | W&B run name. |
| `trainer.wandb.group` | `str | None` | `None` | W&B group. |
| `trainer.wandb.tags` | `list[str] | None` | `None` | W&B tags attached to the run. |
| `trainer.wandb.offline` | `bool` | `False` | Run W&B in offline mode. |

<a id="rl-trainer-bench"></a>
#### `trainer.bench`

Benchmark-mode configuration. When set, ``max_steps`` is forced to 4 and fake data is used.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.bench.output_json` | `Path | None` | `None` | Path to write benchmark results as JSON. If unset, results are only printed to the console. |

<a id="rl-trainer-gc"></a>
#### `trainer.gc`

Garbage collection config. Disables automatic GC and runs deterministic collections every N steps to avoid stragglers. Set to null to use Python's default GC behavior.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.gc.interval` | `int` | `50` | _≥1._ Run garbage collection every N training steps. Disables Python's automatic GC so every rank collects together and one slow rank can't stall the others. |

<a id="rl-trainer-heartbeat"></a>
#### `trainer.heartbeat`

BetterStack heartbeat configuration for monitoring training progress.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.heartbeat.url` | `str` | *required* | URL to send the heartbeat to. |

<a id="rl-trainer-metrics-server"></a>
#### `trainer.metrics_server`

Prometheus metrics server configuration. If set, exposes a ``/metrics`` endpoint for scraping.

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.metrics_server.port` | `int` | `8000` | _≥1, ≤65535._ Port to expose metrics and health endpoints on. |
| `trainer.metrics_server.host` | `str` | `'0.0.0.0'` | Host to bind the server to. |

<a id="rl-trainer-experimental"></a>
#### `trainer.experimental`

<a id="rl-trainer-experimental-token-export"></a>
##### `trainer.experimental.token_export`

Opt-in per-token JSONL export for rollout debugging. When enabled, writes token ids and aligned trainer metrics after each forward pass.

<a id="rl-trainer-loss"></a>
#### `trainer.loss`

Loss config for rl-mode batches. opd and sft batches dispatch to their own loss fns unconditionally and do not read this.

Discriminated union — set `trainer.loss.type` to one of `default`, `custom` and provide the matching sub-fields.

<a id="rl-trainer-loss-default"></a>
##### `trainer.loss.type = "default"` (DefaultLossConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.loss.type` | `'default'` | `'default'` |  |
| `trainer.loss.dppo_mask_low` | `float` | `0.2` | _≥0._ Lower DPPO masking threshold. |
| `trainer.loss.dppo_mask_high` | `float` | `0.2` | _≥0._ Upper DPPO masking threshold. |
| `trainer.loss.adv_tau` | `float` | `1.0` | _≥0._ Temperature for the advantage term. |
| `trainer.loss.kl_tau` | `float` | `0.001` | _≥0._ Temperature for the KL term. |

<a id="rl-trainer-loss-custom"></a>
##### `trainer.loss.type = "custom"` (CustomLossConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.loss.type` | `'custom'` | `'custom'` |  |
| `trainer.loss.import_path` | `str` | *required* | Import path to the loss function (e.g. ``my_module.my_loss``). |
| `trainer.loss.kwargs` | `dict[str, Any]` | `{}` | Kwargs forwarded to the loss function. |

<a id="rl-trainer-optim"></a>
#### `trainer.optim`

Discriminated union — set `trainer.optim.type` to one of `sgd`, `adamw`, `muon`, `sign_sgd` and provide the matching sub-fields.

<a id="rl-trainer-optim-sgd"></a>
##### `trainer.optim.type = "sgd"` (SGDConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `trainer.optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `trainer.optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `trainer.optim.type` | `'sgd'` | `'sgd'` |  |
| `trainer.optim.nesterov` | `bool` | `True` | Use Nesterov momentum. |
| `trainer.optim.momentum` | `float` | `0.9` | SGD momentum factor. |

<a id="rl-trainer-optim-adamw"></a>
##### `trainer.optim.type = "adamw"` (AdamWConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `trainer.optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `trainer.optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `trainer.optim.type` | `'adamw'` | `'adamw'` |  |
| `trainer.optim.betas1` | `float` | `0.9` | _≥0._ Adam first-moment (β1) decay. |
| `trainer.optim.betas2` | `float` | `0.999` | _≥0._ Adam second-moment (β2) decay. |

<a id="rl-trainer-optim-muon"></a>
##### `trainer.optim.type = "muon"` (MuonConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `trainer.optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `trainer.optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `trainer.optim.type` | `'muon'` | `'muon'` |  |
| `trainer.optim.mu` | `float` | `0.95` | _≥0._ Momentum factor for the Muon algorithm. |
| `trainer.optim.betas1` | `float` | `0.9` | _≥0._ β1 for the AdamW/Lion sub-optimizer used on non-Muon params. |
| `trainer.optim.betas2` | `float` | `0.95` | _≥0._ β2 for the AdamW/Lion sub-optimizer used on non-Muon params. |

<a id="rl-trainer-optim-sign-sgd"></a>
##### `trainer.optim.type = "sign_sgd"` (SignSGDConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `trainer.optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `trainer.optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `trainer.optim.type` | `'sign_sgd'` | `'sign_sgd'` |  |

<a id="rl-trainer-scheduler"></a>
#### `trainer.scheduler`

Discriminated union — set `trainer.scheduler.type` to one of `constant`, `linear`, `cosine` and provide the matching sub-fields.

<a id="rl-trainer-scheduler-constant"></a>
##### `trainer.scheduler.type = "constant"` (ConstantSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.scheduler.type` | `'constant'` | `'constant'` |  |

<a id="rl-trainer-scheduler-linear"></a>
##### `trainer.scheduler.type = "linear"` (LinearSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.scheduler.type` | `'linear'` | `'linear'` |  |
| `trainer.scheduler.warmup_steps` | `int` | `10` | _≥0._ Warmup steps for the learning rate scheduler. |
| `trainer.scheduler.decay_steps` | `int` | `10` | _≥0._ Steps to decay the learning rate during the final portion of training. |
| `trainer.scheduler.min_lr` | `float` | `0.0` | _≥0._ Minimum learning rate to converge to. |

<a id="rl-trainer-scheduler-cosine"></a>
##### `trainer.scheduler.type = "cosine"` (CosineSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.scheduler.type` | `'cosine'` | `'cosine'` |  |
| `trainer.scheduler.warmup_steps` | `int` | `10` | _≥0._ Warmup steps for the learning rate scheduler. |
| `trainer.scheduler.min_lr` | `float` | `0.0` | _≥0._ Minimum learning rate to converge to. |

<a id="rl-trainer-weight-broadcast"></a>
#### `trainer.weight_broadcast`

Transport used to broadcast updated weights from trainer to inference.

Discriminated union — set `trainer.weight_broadcast.type` to one of `filesystem`, `nccl` and provide the matching sub-fields.

<a id="rl-trainer-weight-broadcast-filesystem"></a>
##### `trainer.weight_broadcast.type = "filesystem"` (FileSystemWeightBroadcastConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.weight_broadcast.type` | `'filesystem'` | `'filesystem'` |  |
| `trainer.weight_broadcast.save_sharded` | `bool` | `True` | Save the weight checkpoint in sharded format. |
| `trainer.weight_broadcast.save_format` | `'safetensors' | 'torch'` | `'safetensors'` | Weight checkpoint serialization format. |

<a id="rl-trainer-weight-broadcast-nccl"></a>
##### `trainer.weight_broadcast.type = "nccl"` (NCCLWeightBroadcastConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.weight_broadcast.type` | `'nccl'` | `'nccl'` |  |
| `trainer.weight_broadcast.host` | `str` | `'localhost'` | Host for the NCCL broadcast rendezvous. |
| `trainer.weight_broadcast.port` | `int` | `29501` | Port for the NCCL broadcast rendezvous. |
| `trainer.weight_broadcast.timeout` | `int` | `1200` | Timeout in seconds for the NCCL broadcast. |
| `trainer.weight_broadcast.inference_world_size` | `int` | `1` | Number of GPUs used for inference. |
| `trainer.weight_broadcast.quantize_in_weight_transfer` | `bool` | `False` | Use kernel-format FP8 quantized NCCL transfer for weight updates. When disabled, uses default HF checkpoint-format transfer. |

<a id="rl-trainer-rollout-transport"></a>
#### `trainer.rollout_transport`

Transport used to ship rollouts from orchestrator to trainer.

Discriminated union — set `trainer.rollout_transport.type` to one of `filesystem`, `zmq` and provide the matching sub-fields.

<a id="rl-trainer-rollout-transport-filesystem"></a>
##### `trainer.rollout_transport.type = "filesystem"` (FileSystemTransportConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.rollout_transport.type` | `'filesystem'` | `'filesystem'` |  |

<a id="rl-trainer-rollout-transport-zmq"></a>
##### `trainer.rollout_transport.type = "zmq"` (ZMQTransportConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `trainer.rollout_transport.type` | `'zmq'` | `'zmq'` |  |
| `trainer.rollout_transport.host` | `str` | `'localhost'` | Host address for ZMQ transport. |
| `trainer.rollout_transport.port` | `int` | `5555` | Base port for ZMQ transport. |
| `trainer.rollout_transport.hwm` | `int` | `10` | High-water mark (max in-flight messages per ZMQ socket). |

<a id="rl-orchestrator"></a>
### `orchestrator`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.training_mode` | `'rl' | 'opd' | 'sft'` | `'rl'` | Training mode. ``rl``: student generates rollouts, no teacher. ``opd``: student generates rollouts, teacher computes logprobs (teacher_tau > 0). ``sft``: teacher generates rollouts, student inference pool used for evals and weight sync. |
| `orchestrator.advantage` | `DefaultAdvantageConfig | CustomAdvantageConfig | None` | `DefaultAdvantageConfig()` |  |
| `orchestrator.collect_inference_metrics` | `bool` | `True` | Collect inference-server metrics (requires wandb). |
| `orchestrator.output_dir` | `Path` | `'outputs/run_default'` | Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node. |
| `orchestrator.tasks_per_minute` | `int | None` | `None` | _≥1._ Rate limit per environment worker, in tasks per minute. Recommended for sandbox-backed environments to prevent sandbox-not-ready errors during autoscaling. With multiple workers, the effective total rate is ``workers × this value``. None disables rate limiting. |
| `orchestrator.batch_size` | `int | None` | `None` | _≥1._ Samples to train on per step (rollout-based batching). Set this OR ``token_batch_size``. |
| `orchestrator.token_batch_size` | `int | None` | `None` | _≥1._ Tokens to train on per step (token-based batching). Set this OR ``batch_size``. |
| `orchestrator.oversampling_factor` | `float | None` | `None` | _>0._ Rollout-mode batching only. Multiplier used to derive ``max_inflight_rollouts`` from ``batch_size`` when ``max_inflight_rollouts`` is unset. Values below 1.0 intentionally cap in-flight rollout capacity below ``batch_size``. |
| `orchestrator.max_inflight_rollouts` | `int | None` | `None` | _≥1._ Maximum number of rollouts kept in-flight. Required for token-based batching. With ``batch_size`` set, defaults to ``batch_size * oversampling_factor`` (or ``batch_size`` when ``oversampling_factor`` is unset). |
| `orchestrator.group_size` | `int` | `1` | _≥1._ Output sequences returned per example during training. |
| `orchestrator.seq_len` | `int` | `2048` | Training sequence length. Shorter samples are padded; longer samples are truncated. |
| `orchestrator.num_train_workers` | `int` | `1` | _≥1._ Training workers to use. |
| `orchestrator.max_steps` | `int | None` | `None` | Maximum training steps. If None, runs indefinitely. |
| `orchestrator.max_off_policy_steps` | `int` | `8` | _≥0._ Maximum policies allowed to generate a single rollout. Rollouts generated more than ``max_off_policy_steps`` ahead of training are discarded. Higher values yield better throughput at the cost of off-policy noise. |
| `orchestrator.max_async_level` | `int` | `1` | _≥0._ Maximum steps inference can be ahead of training. ``0`` degenerates to synchronous on-policy RL; ``≥1`` overlaps training and inference. |
| `orchestrator.strict_async_level` | `bool` | `False` | Strictly enforce ``max_async_level``. When True, the rollout policy is always exactly ``max_async_level`` steps ahead of training. When False, any policy within ``max_async_level`` steps is allowed (always uses the latest available policy). |
| `orchestrator.bench` | `bool` | `False` | Benchmark mode. Sets ``max_steps`` to 5, ``max_async_level`` to ~∞, and disables W&B. |
| `orchestrator.seed` | `int | None` | `42` | Random seed for the orchestrator. |
| `orchestrator.use_renderer` | `bool` | `True` | Use the renderer-backed TITO client (client-side tokenization via the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package, served by ``/v1/generate``). When True, the ``[orchestrator.renderer]`` block (name / tool_parser / reasoning_parser / pool_size) applies. Default for both text-only and VLM rollouts; VLMs require it. False falls back to MITO (``openai_chat_completions``). |
| `orchestrator.env_install_prerelease` | `bool` | `False` | Allow pre-release versions when installing environments (e.g. ``verifiers>=0.1.12.dev5``). Passes ``--prerelease`` to ``prime env install``. |

<a id="rl-orchestrator-student"></a>
#### `orchestrator.student`

Student rollout participant (model + client) — the model being trained.

<a id="rl-orchestrator-student-model"></a>
##### `orchestrator.student.model`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.student.model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `orchestrator.student.model.trust_remote_code` | `bool` | `False` | Trust remote code when initializing the tokenizer. |

<a id="rl-orchestrator-student-model-vlm"></a>
###### `orchestrator.student.model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.student.model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `orchestrator.student.model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `orchestrator.student.model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="rl-orchestrator-student-model-lora"></a>
###### `orchestrator.student.model.lora`

Per-run LoRA configuration. If None, LoRA is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.student.model.lora.name` | `str | None` | `None` | LoRA adapter name. If None, auto-generated from rank and alpha. |
| `orchestrator.student.model.lora.rank` | `int | None` | `None` | _≥1._ LoRA rank for this run. Must be ≤ trainer's max rank. If None, uses the trainer's rank. |
| `orchestrator.student.model.lora.alpha` | `float | None` | `None` | _≥0._ LoRA alpha for this run. If None, uses the trainer's alpha. |

<a id="rl-orchestrator-student-client"></a>
##### `orchestrator.student.client`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.student.client.timeout` | `int` | `1200` | Request timeout in seconds. |
| `orchestrator.student.client.connect_timeout` | `float` | `30.0` | TCP connect timeout in seconds for inference API requests. |
| `orchestrator.student.client.wait_for_ready_timeout` | `int` | `1800` | Seconds to wait at startup for the inference pool to become ready. Applies to both the static health check and elastic DNS-based discovery. |
| `orchestrator.student.client.base_url` | `list[str]` | `['http://localhost:8000/v1']` | Base URLs for the OpenAI API. With more than one URL, the client round-robins (chat) completion requests across all servers. Ignored when ``elastic`` is set. |
| `orchestrator.student.client.api_key_var` | `str` | `'VLLM_API_KEY'` | Environment variable name containing the API key, resolved via ``os.getenv``. Can be any string when the server is not protected by an API key; the same key is used for every URL. |
| `orchestrator.student.client.headers` | `dict[str, str]` | `{}` | Static headers sent with every request. |
| `orchestrator.student.client.headers_from_env` | `dict[str, str]` | `{}` | Maps HTTP header names to environment variable names; each entry is resolved via ``os.getenv`` and merged into request headers. e.g. ``{"X-Prime-Team-ID": "PRIME_TEAM_ID"}``. |
| `orchestrator.student.client.extra_headers_from_state` | `dict[str, str]` | `{}` | Maps HTTP header names to rollout-state field names. The header value is read from the rollout state dict on every request. e.g. ``{"X-Session-ID": "trajectory_id"}`` enables sticky routing at the inference router. |
| `orchestrator.student.client.skip_model_check` | `bool` | `False` | Skip checking that the model is available in the inference pool. Useful for external APIs or keys that do not expose ``/models``. |
| `orchestrator.student.client.dp_rank_count` | `int` | `1` | _≥1._ Number of data-parallel ranks behind each base URL. When > 1, each URL is expanded into ``dp_rank_count`` logical clients pinned via the ``X-data-parallel-rank`` header, so every request within a rollout hits the same DP engine and reuses KV cache. Auto-set from the inference config when using the RL entrypoint. |
| `orchestrator.student.client.admin_base_url` | `list[str] | None` | `None` | Separate base URLs for admin operations (weight updates, health checks). When set, admin clients bypass routers and hit each server directly — used in disaggregated P/D deployments where the router must not handle admin traffic. |
| `orchestrator.student.client.router_url` | `str | None` | `None` | vllm-router URL for load-aware inference routing. With elastic mode, inference requests go through the router while admin ops still hit discovered pods directly. |

<a id="rl-orchestrator-student-client-elastic"></a>
###### `orchestrator.student.client.elastic`

Elastic inference pool config for DNS-based service discovery. When set, ``base_url`` is ignored and inference servers are discovered dynamically via DNS.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.student.client.elastic.hostname` | `str` | *required* | DNS hostname that resolves to inference server IPs. |
| `orchestrator.student.client.elastic.port` | `int` | `8000` | Port that inference servers listen on. |
| `orchestrator.student.client.elastic.sync_interval` | `float` | `5.0` | Seconds between server discovery checks. |

<a id="rl-orchestrator-teacher"></a>
#### `orchestrator.teacher`

Teacher rollout participant (model + client). Role depends on ``training_mode``: ``opd`` — teacher computes logprobs; ``sft`` — teacher generates rollouts.

<a id="rl-orchestrator-teacher-model"></a>
##### `orchestrator.teacher.model`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.teacher.model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `orchestrator.teacher.model.trust_remote_code` | `bool` | `False` | Trust remote code when initializing the tokenizer. |

<a id="rl-orchestrator-teacher-model-vlm"></a>
###### `orchestrator.teacher.model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.teacher.model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `orchestrator.teacher.model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `orchestrator.teacher.model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="rl-orchestrator-teacher-model-lora"></a>
###### `orchestrator.teacher.model.lora`

Per-run LoRA configuration. If None, LoRA is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.teacher.model.lora.name` | `str | None` | `None` | LoRA adapter name. If None, auto-generated from rank and alpha. |
| `orchestrator.teacher.model.lora.rank` | `int | None` | `None` | _≥1._ LoRA rank for this run. Must be ≤ trainer's max rank. If None, uses the trainer's rank. |
| `orchestrator.teacher.model.lora.alpha` | `float | None` | `None` | _≥0._ LoRA alpha for this run. If None, uses the trainer's alpha. |

<a id="rl-orchestrator-teacher-client"></a>
##### `orchestrator.teacher.client`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.teacher.client.timeout` | `int` | `1200` | Request timeout in seconds. |
| `orchestrator.teacher.client.connect_timeout` | `float` | `30.0` | TCP connect timeout in seconds for inference API requests. |
| `orchestrator.teacher.client.wait_for_ready_timeout` | `int` | `1800` | Seconds to wait at startup for the inference pool to become ready. Applies to both the static health check and elastic DNS-based discovery. |
| `orchestrator.teacher.client.base_url` | `list[str]` | `['http://localhost:8000/v1']` | Base URLs for the OpenAI API. With more than one URL, the client round-robins (chat) completion requests across all servers. Ignored when ``elastic`` is set. |
| `orchestrator.teacher.client.api_key_var` | `str` | `'VLLM_API_KEY'` | Environment variable name containing the API key, resolved via ``os.getenv``. Can be any string when the server is not protected by an API key; the same key is used for every URL. |
| `orchestrator.teacher.client.headers` | `dict[str, str]` | `{}` | Static headers sent with every request. |
| `orchestrator.teacher.client.headers_from_env` | `dict[str, str]` | `{}` | Maps HTTP header names to environment variable names; each entry is resolved via ``os.getenv`` and merged into request headers. e.g. ``{"X-Prime-Team-ID": "PRIME_TEAM_ID"}``. |
| `orchestrator.teacher.client.extra_headers_from_state` | `dict[str, str]` | `{}` | Maps HTTP header names to rollout-state field names. The header value is read from the rollout state dict on every request. e.g. ``{"X-Session-ID": "trajectory_id"}`` enables sticky routing at the inference router. |
| `orchestrator.teacher.client.skip_model_check` | `bool` | `False` | Skip checking that the model is available in the inference pool. Useful for external APIs or keys that do not expose ``/models``. |
| `orchestrator.teacher.client.dp_rank_count` | `int` | `1` | _≥1._ Number of data-parallel ranks behind each base URL. When > 1, each URL is expanded into ``dp_rank_count`` logical clients pinned via the ``X-data-parallel-rank`` header, so every request within a rollout hits the same DP engine and reuses KV cache. Auto-set from the inference config when using the RL entrypoint. |
| `orchestrator.teacher.client.admin_base_url` | `list[str] | None` | `None` | Separate base URLs for admin operations (weight updates, health checks). When set, admin clients bypass routers and hit each server directly — used in disaggregated P/D deployments where the router must not handle admin traffic. |
| `orchestrator.teacher.client.router_url` | `str | None` | `None` | vllm-router URL for load-aware inference routing. With elastic mode, inference requests go through the router while admin ops still hit discovered pods directly. |

<a id="rl-orchestrator-teacher-client-elastic"></a>
###### `orchestrator.teacher.client.elastic`

Elastic inference pool config for DNS-based service discovery. When set, ``base_url`` is ignored and inference servers are discovered dynamically via DNS.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.teacher.client.elastic.hostname` | `str` | *required* | DNS hostname that resolves to inference server IPs. |
| `orchestrator.teacher.client.elastic.port` | `int` | `8000` | Port that inference servers listen on. |
| `orchestrator.teacher.client.elastic.sync_interval` | `float` | `5.0` | Seconds between server discovery checks. |

<a id="rl-orchestrator-train"></a>
#### `orchestrator.train`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.train.num_workers` | `int | 'auto'` | `'auto'` | Default worker processes for env servers. Can be overridden per env. |
| `orchestrator.train.max_retries` | `int` | `3` | _≥0._ Default retries for failed rollouts. Can be overridden per env. |

<a id="rl-orchestrator-train-sampling"></a>
##### `orchestrator.train.sampling`

Shared training sampling configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.train.sampling.temperature` | `float` | `1.0` | _≥0._ Sampling temperature. |
| `orchestrator.train.sampling.repetition_penalty` | `float` | `1.0` | _≥0._ Repetition penalty. Values > 1.0 discourage repetition, < 1.0 encourage it, 1.0 disables. |
| `orchestrator.train.sampling.max_completion_tokens` | `int | None` | `None` | Maximum output tokens per turn. If None, generates until max context length or EOS. |
| `orchestrator.train.sampling.min_tokens` | `int` | `0` | _≥0._ Minimum output tokens per sequence. |
| `orchestrator.train.sampling.seed` | `int | None` | `None` | Random seed for sampling. If None, no seeding is used. |
| `orchestrator.train.sampling.extra_body` | `dict[str, Any]` | `{}` | Extra body forwarded with each request to the inference server. |

<a id="rl-orchestrator-train-env"></a>
##### `orchestrator.train.env.<n>` (list item)

Training environments.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.train.env.<n>.id` | `str` | `'reverse-text'` | Registered [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment ID (e.g. ``math-env``, ``primeintellect/math-env``). May include an ``@version`` suffix for installation. |
| `orchestrator.train.env.<n>.name` | `str | None` | `None` | Display name for this environment in logs, metrics, and buffer keys. Defaults to the ``id`` without ``@version``. Must be unique across all envs in the same group. |
| `orchestrator.train.env.<n>.args` | `dict` | `{}` | Keyword arguments forwarded to ``vf.load_environment``. See the environment's docstring for accepted args. |
| `orchestrator.train.env.<n>.extra_env_kwargs` | `dict[str, Any]` | `{}` | Extra kwargs passed to the env (e.g. ``seq_len``, ``max_total_completion_tokens``). Auto-populated by the orchestrator; user overrides are generally discouraged. The main use case is matching ``extra_env_kwargs`` when running an env in an isolated environment server. |
| `orchestrator.train.env.<n>.address` | `str | None` | `None` | ZMQ address of an external env server (e.g. ``tcp://host:5000``). When set, the orchestrator connects to this server instead of spawning one; when None, a subprocess env server is spawned automatically. |
| `orchestrator.train.env.<n>.num_workers` | `int | 'auto'` | `'auto'` | Worker processes for the spawned env server. ``auto`` scales to 1 worker per 256 concurrent rollouts. Ignored when ``address`` is set. |
| `orchestrator.train.env.<n>.ratio` | `float | None` | `None` | _>0._ Sampling weight for this environment in the buffer. When None for all envs, samples uniformly across all available problems. When set, must be set on all envs — values are relative weights normalized to probabilities (e.g. [1, 1] and [0.5, 0.5] are equivalent). |
| `orchestrator.train.env.<n>.max_retries` | `int` | `3` | _≥0._ Times the env server retries a failed rollout before returning an error. |
| `orchestrator.train.env.<n>.max_total_completion_tokens` | `int` | `-1` | Maximum total completion tokens across all turns in a multi-turn rollout. ``-1`` disables. Auto-populated into ``extra_env_kwargs``. |
| `orchestrator.train.env.<n>.timeout` | `float | None` | `None` | Per-rollout wall-clock timeout in seconds. None disables. |
| `orchestrator.train.env.<n>.state_columns` | `list[str]` | `[]` | Extra ``State`` fields to persist into the saved rollout records (in addition to the always-saved ``trajectory`` and ``sampling_args``). Values must be JSON-serializable. |

<a id="rl-orchestrator-train-env-sampling"></a>
###### `orchestrator.train.env.<n>.sampling`

Per-env sampling overrides. Unset fields inherit from the group-level train sampling config.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.train.env.<n>.sampling.temperature` | `float` | `1.0` | _≥0._ Sampling temperature. |
| `orchestrator.train.env.<n>.sampling.repetition_penalty` | `float` | `1.0` | _≥0._ Repetition penalty. Values > 1.0 discourage repetition, < 1.0 encourage it, 1.0 disables. |
| `orchestrator.train.env.<n>.sampling.max_completion_tokens` | `int | None` | `None` | Maximum output tokens per turn. If None, generates until max context length or EOS. |
| `orchestrator.train.env.<n>.sampling.min_tokens` | `int` | `0` | _≥0._ Minimum output tokens per sequence. |
| `orchestrator.train.env.<n>.sampling.seed` | `int | None` | `None` | Random seed for sampling. If None, no seeding is used. |
| `orchestrator.train.env.<n>.sampling.extra_body` | `dict[str, Any]` | `{}` | Extra body forwarded with each request to the inference server. |

<a id="rl-orchestrator-tokenizer"></a>
#### `orchestrator.tokenizer`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.tokenizer.name` | `str | None` | `None` | Tokenizer name or path. If None, the model's default tokenizer is used. |
| `orchestrator.tokenizer.trust_remote_code` | `bool | None` | `None` | Trust remote code when initializing the tokenizer. If None, inherits the model's ``trust_remote_code`` setting. |
| `orchestrator.tokenizer.chat_template` | `str | None` | `None` | Chat template for the tokenizer. Either a Jinja2 template string or a path to a template file. If None, the tokenizer's default chat template is used. |

<a id="rl-orchestrator-renderer"></a>
#### `orchestrator.renderer`

Client-side renderer configuration. Only consumed when ``use_renderer=true``.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.renderer.name` | `str` | `'auto'` | Renderer used for chat-template tokenization. One of: ``auto`` (detect from tokenizer), ``qwen3``, ``qwen3_vl``, ``qwen3.5``, ``glm5``, ``glm4.5``, ``minimax-m2``, ``deepseek_v3``, ``kimi_k2``, ``kimi_k25``, ``nemotron3``, ``gpt_oss``, ``default``. |
| `orchestrator.renderer.tool_parser` | `str | None` | `None` | Tool parser from [`renderers.parsers`](https://github.com/PrimeIntellect-ai/renderers). Only consumed by DefaultRenderer; model-specific renderers bake their own parsing in. Options: ``qwen3``, ``qwen3.5``, ``glm``, ``deepseek_v3``. |
| `orchestrator.renderer.reasoning_parser` | `str | None` | `None` | Reasoning parser from [`renderers.parsers`](https://github.com/PrimeIntellect-ai/renderers). Only consumed by DefaultRenderer. Options: ``think``. |
| `orchestrator.renderer.pool_size` | `int | None` | `None` | _≥1._ Number of renderer slots shared across concurrent rollouts. Bump for long multi-turn prompts where client-side jinja tokenization serializes. |
| `orchestrator.renderer.preserve_all_thinking` | `bool` | `False` | Re-emit every past-assistant turn's ``reasoning_content`` between ``<think>``/``</think>`` (or the model's equivalent), even when the chat template would drop it. Strict superset of preserve_thinking_between_tool_calls. |
| `orchestrator.renderer.preserve_thinking_between_tool_calls` | `bool` | `False` | Preserve past-assistant ``reasoning_content`` only inside the current tool cycle — the contiguous assistant→tool→…→assistant block after the most recent user message, when that block contains at least one tool response. A new user turn closes the block. |

<a id="rl-orchestrator-optim"></a>
#### `orchestrator.optim`

Per-run optimizer configuration for multi-run training.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.optim.lr` | `float` | `0.0001` | _≥0._ Learning rate for this run (per-run override for multi-run training). |

<a id="rl-orchestrator-eval"></a>
#### `orchestrator.eval`

Evaluation configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.eval.num_examples` | `int` | `-1` | Default eval examples per environment. ``-1`` uses all. Can be overridden per env. |
| `orchestrator.eval.group_size` | `int` | `1` | _≥1._ Default rollouts per example. Can be overridden per env. |
| `orchestrator.eval.num_workers` | `int | 'auto'` | `'auto'` | Default worker processes for env servers. Can be overridden per env. |
| `orchestrator.eval.max_retries` | `int` | `3` | _≥0._ Default retries for failed rollouts. Can be overridden per env. |
| `orchestrator.eval.interval` | `int` | `100` | _≥1._ Step interval at which to evaluate the model. |
| `orchestrator.eval.eval_base_model` | `bool` | `True` | Evaluate the base model we are training on. |
| `orchestrator.eval.skip_eval_on_resume` | `bool` | `True` | When resuming the orchestrator from a checkpoint, skip the (potentially redundant) online eval that would otherwise run immediately at the resumed step. |
| `orchestrator.eval.cancel_inflight_rollouts_on_eval` | `bool` | `False` | Cancel in-flight training rollouts before starting online evals. Avoids congestion (no training + eval rollouts at the same time) at the cost of slower training steps as the pipeline has to refill after each eval. |

<a id="rl-orchestrator-eval-sampling"></a>
##### `orchestrator.eval.sampling`

Shared eval sampling configuration; can differ from training sampling.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.eval.sampling.temperature` | `float | None` | `None` | _≥0._ Sampling temperature. None defers to the inference server default. |
| `orchestrator.eval.sampling.repetition_penalty` | `float | None` | `None` | _≥0._ Repetition penalty. None defers to the inference server default. |
| `orchestrator.eval.sampling.top_p` | `float | None` | `None` | Nucleus sampling threshold. None defers to the inference server default. |
| `orchestrator.eval.sampling.top_k` | `int | None` | `None` | Top-k sampling. None defers to the inference server default. |
| `orchestrator.eval.sampling.min_p` | `float | None` | `None` | _≥0._ Min-p sampling threshold. None defers to the inference server default. |
| `orchestrator.eval.sampling.max_completion_tokens` | `int | None` | `None` | Maximum output tokens per turn. None defers to the inference server default. |
| `orchestrator.eval.sampling.min_tokens` | `int | None` | `None` | _≥0._ Minimum output tokens per sequence. None defers to the inference server default. |
| `orchestrator.eval.sampling.reasoning_effort` | `'minimal' | 'low' | 'medium' | 'high' | None` | `None` | Reasoning effort constraint for reasoning models. |
| `orchestrator.eval.sampling.seed` | `int | None` | `None` | Random seed for sampling. None means no seeding. |
| `orchestrator.eval.sampling.extra_body` | `dict[str, Any]` | `{}` | Extra body parameters forwarded to the inference server. |

<a id="rl-orchestrator-eval-env"></a>
##### `orchestrator.eval.env.<n>` (list item)

Evaluation environments.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.eval.env.<n>.id` | `str` | `'reverse-text'` | Registered [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment ID (e.g. ``math-env``, ``primeintellect/math-env``). May include an ``@version`` suffix for installation. |
| `orchestrator.eval.env.<n>.name` | `str | None` | `None` | Display name for this environment in logs, metrics, and buffer keys. Defaults to the ``id`` without ``@version``. Must be unique across all envs in the same group. |
| `orchestrator.eval.env.<n>.args` | `dict` | `{}` | Keyword arguments forwarded to ``vf.load_environment``. See the environment's docstring for accepted args. |
| `orchestrator.eval.env.<n>.extra_env_kwargs` | `dict[str, Any]` | `{}` | Extra kwargs passed to the env (e.g. ``seq_len``, ``max_total_completion_tokens``). Auto-populated by the orchestrator; user overrides are generally discouraged. The main use case is matching ``extra_env_kwargs`` when running an env in an isolated environment server. |
| `orchestrator.eval.env.<n>.address` | `str | None` | `None` | ZMQ address of an external env server (e.g. ``tcp://host:5000``). When set, the orchestrator connects to this server instead of spawning one; when None, a subprocess env server is spawned automatically. |
| `orchestrator.eval.env.<n>.num_workers` | `int | 'auto'` | `'auto'` | Worker processes for the spawned env server. ``auto`` scales to 1 worker per 256 concurrent rollouts. Ignored when ``address`` is set. |
| `orchestrator.eval.env.<n>.ratio` | `float | None` | `None` | _>0._ Sampling weight for this environment in the buffer. When None for all envs, samples uniformly across all available problems. When set, must be set on all envs — values are relative weights normalized to probabilities (e.g. [1, 1] and [0.5, 0.5] are equivalent). |
| `orchestrator.eval.env.<n>.max_retries` | `int` | `3` | _≥0._ Times the env server retries a failed rollout before returning an error. |
| `orchestrator.eval.env.<n>.max_total_completion_tokens` | `int` | `-1` | Maximum total completion tokens across all turns in a multi-turn rollout. ``-1`` disables. Auto-populated into ``extra_env_kwargs``. |
| `orchestrator.eval.env.<n>.timeout` | `float | None` | `None` | Per-rollout wall-clock timeout in seconds. None disables. |
| `orchestrator.eval.env.<n>.state_columns` | `list[str]` | `[]` | Extra ``State`` fields to persist into the saved rollout records (in addition to the always-saved ``trajectory`` and ``sampling_args``). Values must be JSON-serializable. |
| `orchestrator.eval.env.<n>.num_examples` | `int` | `-1` | Eval examples to sample from the dataset. ``-1`` uses all available examples. |
| `orchestrator.eval.env.<n>.group_size` | `int` | `1` | _≥1._ Rollouts generated per example. Used for pass@k estimation (e.g. ``group_size=8`` enables pass@1 through pass@8). |
| `orchestrator.eval.env.<n>.interval` | `int` | `100` | _≥1._ Per-env eval interval. If unset, inherits from the group-level eval interval. |

<a id="rl-orchestrator-eval-env-sampling"></a>
###### `orchestrator.eval.env.<n>.sampling`

Per-env sampling overrides. Unset fields inherit from the group-level eval sampling config.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.eval.env.<n>.sampling.temperature` | `float | None` | `None` | _≥0._ Sampling temperature. None defers to the inference server default. |
| `orchestrator.eval.env.<n>.sampling.repetition_penalty` | `float | None` | `None` | _≥0._ Repetition penalty. None defers to the inference server default. |
| `orchestrator.eval.env.<n>.sampling.top_p` | `float | None` | `None` | Nucleus sampling threshold. None defers to the inference server default. |
| `orchestrator.eval.env.<n>.sampling.top_k` | `int | None` | `None` | Top-k sampling. None defers to the inference server default. |
| `orchestrator.eval.env.<n>.sampling.min_p` | `float | None` | `None` | _≥0._ Min-p sampling threshold. None defers to the inference server default. |
| `orchestrator.eval.env.<n>.sampling.max_completion_tokens` | `int | None` | `None` | Maximum output tokens per turn. None defers to the inference server default. |
| `orchestrator.eval.env.<n>.sampling.min_tokens` | `int | None` | `None` | _≥0._ Minimum output tokens per sequence. None defers to the inference server default. |
| `orchestrator.eval.env.<n>.sampling.reasoning_effort` | `'minimal' | 'low' | 'medium' | 'high' | None` | `None` | Reasoning effort constraint for reasoning models. |
| `orchestrator.eval.env.<n>.sampling.seed` | `int | None` | `None` | Random seed for sampling. None means no seeding. |
| `orchestrator.eval.env.<n>.sampling.extra_body` | `dict[str, Any]` | `{}` | Extra body parameters forwarded to the inference server. |

<a id="rl-orchestrator-buffer"></a>
#### `orchestrator.buffer`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.buffer.seed` | `int | None` | `None` | Random seed for the buffer. When set, sampling from the buffer is deterministic. |
| `orchestrator.buffer.easy_threshold` | `float | None` | `None` | Average-reward threshold above which a problem is classified ``easy``. |
| `orchestrator.buffer.hard_threshold` | `float | None` | `None` | Average-reward threshold below which a problem is classified ``hard``. |
| `orchestrator.buffer.easy_fraction` | `float` | `0.0` | _≥0, ≤1._ Fraction of easy problems to convert to ``normal`` when resuming or starting training. Only problems with difficulty ``normal`` are sampled. |
| `orchestrator.buffer.hard_fraction` | `float` | `0.0` | _≥0, ≤1._ Fraction of hard problems to convert to ``normal`` when resuming or starting training. Only problems with difficulty ``normal`` are sampled. |
| `orchestrator.buffer.online_difficulty_filtering` | `bool` | `False` | Filter rollouts based on difficulty. When True, rollouts with average reward 0.0 or 1.0 are not added to the buffer. |
| `orchestrator.buffer.hash_keys` | `list[str]` | `['env_name', 'prompt']` | _len ≥ 1._ Keys used to compute example hashes. Used to match examples from buffer checkpoints and determine buffer resume behavior. |

<a id="rl-orchestrator-log"></a>
#### `orchestrator.log`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.log.level` | `str` | `'info'` | Log level for the process. Defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``. |
| `orchestrator.log.vf_level` | `str` | `'info'` | Log level for the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) package. Defaults to ``$PRIME_VF_LOG_LEVEL`` if set, else ``info``. |
| `orchestrator.log.json_logging` | `bool` | `False` | Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.). |
| `orchestrator.log.log_data` | `bool` | `False` | Log the first data sample at startup. |

<a id="rl-orchestrator-wandb"></a>
#### `orchestrator.wandb`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.wandb.project` | `str` | `'prime-rl'` | W&B project to log to. |
| `orchestrator.wandb.entity` | `str | None` | `None` | W&B entity to log to. |
| `orchestrator.wandb.name` | `str | None` | `None` | W&B run name. |
| `orchestrator.wandb.group` | `str | None` | `None` | W&B group. |
| `orchestrator.wandb.tags` | `list[str] | None` | `None` | W&B tags attached to the run. |
| `orchestrator.wandb.offline` | `bool` | `False` | Run W&B in offline mode. |

<a id="rl-orchestrator-wandb-log-extras"></a>
##### `orchestrator.wandb.log_extras`

Extras logging configuration. If None, no extras are logged.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.wandb.log_extras.samples` | `bool` | `True` | Log prompt/response samples. |
| `orchestrator.wandb.log_extras.distributions` | `bool` | `True` | Log distributions (rewards, advantages, etc.). |
| `orchestrator.wandb.log_extras.interval` | `int` | `10` | _≥1._ Step interval between extras logs. |
| `orchestrator.wandb.log_extras.sample_ratio` | `float | None` | `None` | _≥0.0, ≤1.0._ Fraction of rollouts to log per step. The effective cap is ``len(rollouts) * sample_ratio``; 1.0 = all, 0.5 = half, 0.0 = none. |

<a id="rl-orchestrator-prime-monitor"></a>
#### `orchestrator.prime_monitor`

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.prime_monitor.base_url` | `str` | `'https://api.primeintellect.ai/api/v1/rft'` | Base URL for the Prime Intellect monitoring API. |
| `orchestrator.prime_monitor.api_key_var` | `str` | `'PRIME_API_KEY'` | Environment variable name containing the Prime Intellect API key, resolved via ``os.getenv``. |
| `orchestrator.prime_monitor.run_name` | `str | None` | `None` | Run name shown on the platform. Defaults to the W&B run name when set, otherwise the platform auto-generates one. |
| `orchestrator.prime_monitor.team_id` | `str | None` | `None` | Team ID to associate the run with. |
| `orchestrator.prime_monitor.frontend_url` | `str | None` | `None` | Frontend base URL used for the dashboard link printed after registration. Defaults to the Prime CLI frontend URL when unset. |

<a id="rl-orchestrator-prime-monitor-log-extras"></a>
##### `orchestrator.prime_monitor.log_extras`

Extras logging configuration. If None, no extras are logged.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.prime_monitor.log_extras.samples` | `bool` | `True` | Log prompt/response samples. |
| `orchestrator.prime_monitor.log_extras.distributions` | `bool` | `True` | Log distributions (rewards, advantages, etc.). |
| `orchestrator.prime_monitor.log_extras.interval` | `int` | `10` | _≥1._ Step interval between extras logs. |
| `orchestrator.prime_monitor.log_extras.sample_ratio` | `float | None` | `None` | _≥0.0, ≤1.0._ Fraction of rollouts to log per step. The effective cap is ``len(rollouts) * sample_ratio``; 1.0 = all, 0.5 = half, 0.0 = none. |

<a id="rl-orchestrator-ckpt"></a>
#### `orchestrator.ckpt`

Checkpoint configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.ckpt.interval` | `int | None` | `None` | _≥1._ Step interval at which to save the orchestrator checkpoint. |
| `orchestrator.ckpt.resume_step` | `int | None` | `None` | _≥-1._ Step to resume the orchestrator from. None starts from scratch; ``-1`` resumes from the latest checkpoint available. |
| `orchestrator.ckpt.wait_for_weights_timeout` | `int | None` | `None` | _≥1._ When resuming, wait up to this many seconds for the weight directory to appear. Useful when the orchestrator restarts while the trainer is still saving weights. If None, fail immediately when weights are not found. |
| `orchestrator.ckpt.keep_last` | `int | None` | `None` | _≥1._ Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency. |
| `orchestrator.ckpt.keep_interval` | `int | None` | `None` | _≥1._ Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping. |
| `orchestrator.ckpt.skip_progress` | `bool` | `False` | Skip loading the progress from checkpoint. |
| `orchestrator.ckpt.skip_buffer` | `bool` | `False` | Skip loading the buffer from checkpoint. |

<a id="rl-orchestrator-heartbeat"></a>
#### `orchestrator.heartbeat`

BetterStack heartbeat configuration for monitoring training progress.

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.heartbeat.url` | `str` | *required* | URL to send the heartbeat to. |

<a id="rl-orchestrator-experimental"></a>
#### `orchestrator.experimental`

<a id="rl-orchestrator-filters"></a>
#### `orchestrator.filters.<n>` (list item)

Rollout filters. Each filter can ``monitor`` (default) or ``enforce`` (skip rollouts).

Discriminated list-item union — set `orchestrator.filters.<n>.type` to one of `gibberish`, `repetition`, `zero_advantage` and provide the matching sub-fields.

<a id="rl-orchestrator-filters-gibberish"></a>
##### `orchestrator.filters.<n>.type = "gibberish"` (GibberishFilterConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.filters.<n>.type` | `'gibberish'` | `'gibberish'` |  |
| `orchestrator.filters.<n>.enforce` | `bool` | `False` | When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics. |
| `orchestrator.filters.<n>.token_id_threshold` | `int` | `100000` | Token IDs above this are candidates for gibberish. BPE tokens are sorted by merge order. |
| `orchestrator.filters.<n>.logprob_offset` | `float` | `2.0` | Offset from uniform-distribution logprob. Threshold = ``-log(vocab_size) - logprob_offset``. |

<a id="rl-orchestrator-filters-repetition"></a>
##### `orchestrator.filters.<n>.type = "repetition"` (RepetitionFilterConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.filters.<n>.type` | `'repetition'` | `'repetition'` |  |
| `orchestrator.filters.<n>.enforce` | `bool` | `False` | When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics. |
| `orchestrator.filters.<n>.window` | `int` | `3000` | _≥1._ Consecutive high-probability steps required to flag the rollout. |
| `orchestrator.filters.<n>.prob_threshold` | `float` | `0.99` | _>0, ≤1._ Tokens sampled with probability above this are considered repetitive. Consecutive such tokens count toward the window. |

<a id="rl-orchestrator-filters-zero-advantage"></a>
##### `orchestrator.filters.<n>.type = "zero_advantage"` (ZeroAdvantageFilterConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.filters.<n>.type` | `'zero_advantage'` | `'zero_advantage'` |  |
| `orchestrator.filters.<n>.enforce` | `bool` | `True` | When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics. |

<a id="rl-orchestrator-weight-broadcast"></a>
#### `orchestrator.weight_broadcast`

Transport used to receive updated weights from the trainer.

Discriminated union — set `orchestrator.weight_broadcast.type` to one of `filesystem`, `nccl` and provide the matching sub-fields.

<a id="rl-orchestrator-weight-broadcast-filesystem"></a>
##### `orchestrator.weight_broadcast.type = "filesystem"` (FileSystemWeightBroadcastConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.weight_broadcast.type` | `'filesystem'` | `'filesystem'` |  |

<a id="rl-orchestrator-weight-broadcast-nccl"></a>
##### `orchestrator.weight_broadcast.type = "nccl"` (NCCLWeightBroadcastConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.weight_broadcast.type` | `'nccl'` | `'nccl'` |  |
| `orchestrator.weight_broadcast.host` | `str` | `'localhost'` | Host for the NCCL broadcast rendezvous. |
| `orchestrator.weight_broadcast.port` | `int` | `29501` | Port for the NCCL broadcast rendezvous. |
| `orchestrator.weight_broadcast.timeout` | `int` | `1200` | Timeout in seconds for the NCCL broadcast. |
| `orchestrator.weight_broadcast.quantize_in_weight_transfer` | `bool` | `False` | Use kernel-format FP8 quantized NCCL transfer for weight updates. |
| `orchestrator.weight_broadcast.inference_world_size` | `int` | `1` | _≥1._ Total inference GPUs across all servers. Used by ``init_nccl_broadcast`` to compute per-server rank offsets. |

<a id="rl-orchestrator-rollout-transport"></a>
#### `orchestrator.rollout_transport`

Transport used to ship rollouts from orchestrator to trainer.

Discriminated union — set `orchestrator.rollout_transport.type` to one of `filesystem`, `zmq` and provide the matching sub-fields.

<a id="rl-orchestrator-rollout-transport-filesystem"></a>
##### `orchestrator.rollout_transport.type = "filesystem"` (FileSystemTransportConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.rollout_transport.type` | `'filesystem'` | `'filesystem'` |  |

<a id="rl-orchestrator-rollout-transport-zmq"></a>
##### `orchestrator.rollout_transport.type = "zmq"` (ZMQTransportConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `orchestrator.rollout_transport.type` | `'zmq'` | `'zmq'` |  |
| `orchestrator.rollout_transport.host` | `str` | `'localhost'` | Host address for ZMQ transport. |
| `orchestrator.rollout_transport.port` | `int` | `5555` | Base port for ZMQ transport. |
| `orchestrator.rollout_transport.hwm` | `int` | `10` | High-water mark (max in-flight messages per ZMQ socket). |

<a id="rl-inference"></a>
### `inference`

Inference server configuration. If None, the rl entrypoint will not start an inference server (useful for elastic inference pools or manually started servers).

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.enable_lora` | `bool` | `False` | Enable LoRA. Forwarded as ``--enable-lora``. |
| `inference.max_loras` | `int` | `8` | Maximum number of LoRAs. Forwarded as ``--max-loras``. |
| `inference.max_cpu_loras` | `int` | `100` | Maximum number of LoRAs on CPU. Forwarded as ``--max-cpu-loras``. |
| `inference.max_lora_rank` | `int | None` | `None` | Maximum LoRA rank. Forwarded as ``--max-lora-rank``. |
| `inference.lora_target_modules` | `list[str] | None` | `None` | LoRA target modules. Forwarded as ``--lora-target-modules``. |
| `inference.enable_prefix_caching` | `bool | None` | `None` | Enable prefix caching. Forwarded as ``--enable-prefix-caching``. |
| `inference.gpu_memory_utilization` | `float` | `0.9` | GPU memory utilization. Forwarded as ``--gpu-memory-utilization``. |
| `inference.api_server_count` | `int` | `1` | _≥0._ API servers to run. Forwarded as ``--api-server-count``. Set to 0 for headless mode. |
| `inference.data_parallel_size_local` | `int | None` | `None` | _≥1._ Data parallel replicas to run on this node. Forwarded as ``--data-parallel-size-local``. |
| `inference.data_parallel_rpc_port` | `int` | `13345` | _≥1, ≤65535._ RPC port for data parallel communication. Forwarded as ``--data-parallel-rpc-port``. |
| `inference.seed` | `int` | `0` | Seed the inference components. Forwarded as ``--seed``. |
| `inference.enable_expert_parallel` | `bool` | `False` | Enable expert parallelism for MoE models. Forwarded as ``--enable-expert-parallel``. |
| `inference.all2all_backend` | `'allgather_reducescatter' | 'deepep_high_throughput' | 'deepep_low_latency' | 'flashinfer_nvlink_one_sided' | 'flashinfer_nvlink_two_sided'` | `'allgather_reducescatter'` | All-to-all backend for expert-parallel communication. Forwarded as ``--all2all-backend``. |
| `inference.enable_eplb` | `bool` | `False` | Enable expert parallel load balancer (EPLB). Forwarded as ``--enable-eplb``. |
| `inference.enable_dbo` | `bool` | `False` | Enable dual batch overlap (DBO). Forwarded as ``--enable-dbo``. |
| `inference.use_deep_gemm` | `bool` | `False` | Force DeepGEMM FP8 kernels via ``VLLM_USE_DEEP_GEMM=1``. Only works with per-tensor FP8 quantization (e.g. GLM-5-FP8). |
| `inference.enable_return_routed_experts` | `bool` | `False` | Return routed experts in responses. Forwarded as ``--enable-return-routed-experts``. |
| `inference.enable_fp32_lm_head` | `bool` | `False` | Run the lm_head projection in fp32 via a native bf16×bf16 → fp32 GEMM (``torch.mm`` with ``out_dtype=torch.float32``). Stabilizes logprob precision under FP8/bf16 inference, matching SGLang's ``--enable-fp32-lm-head``. Implemented as a monkey-patch over vLLM's LogitsProcessor, activated by setting ``additional_config["fp32_lm_head"] = True`` on the vLLM config. |
| `inference.vllm_extra` | `dict[str, Any]` | `{}` | Extra arguments forwarded to vLLM. Applied as attributes on the vLLM namespace after config translation. |
| `inference.output_dir` | `Path` | `'outputs'` | Directory for SLURM logs and generated scripts. |
| `inference.dry_run` | `bool` | `False` | Only validate and dump resolved configs, then exit early. |

<a id="rl-inference-server"></a>
#### `inference.server`

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.server.host` | `str | None` | `None` | Host to bind to. |
| `inference.server.port` | `int` | `8000` | Port to bind to. |
| `inference.server.liveness_timeout_seconds` | `float` | `30.0` | _>0._ Timeout in seconds for the ``/liveness`` endpoint's internal vLLM worker RPC. With Kubernetes liveness probes, keep the probe ``timeoutSeconds`` at least this high. |

<a id="rl-inference-model"></a>
#### `inference.model`

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `inference.model.trust_remote_code` | `bool` | `False` | Trust remote code. Forwarded to vLLM engine init. |
| `inference.model.dtype` | `'auto' | 'float16' | 'bfloat16' | 'float32'` | `'auto'` | dtype for model weights and activations. ``auto`` uses FP16 for FP32/FP16 models and BF16 for BF16 models. Forwarded as ``--dtype``. |
| `inference.model.max_model_len` | `int | None` | `None` | Maximum model context length. If None, uses the model config's value. Forwarded as ``--max-model-len``. |
| `inference.model.enforce_eager` | `bool` | `False` | Enforce eager mode. When False, PyTorch eager and cuda graphs run hybrid for maximum performance. Forwarded as ``--enforce-eager``. |
| `inference.model.chat_template` | `str | None` | `None` | Chat template — a Jinja2 template string or path to a template file. Forwarded as ``--chat-template``. If None, uses the model's default. |
| `inference.model.tool_call_parser` | `str | None` | `'auto'` | Tool-call parser. Forwarded as ``--tool-call-parser``. Set to ``"auto"`` (default) to detect from the model name, or ``None`` to disable. |
| `inference.model.reasoning_parser` | `str | None` | `'auto'` | Parser for extracting reasoning content from model outputs. Forwarded as ``--reasoning-parser``. Set to ``"auto"`` (default) to detect from the model name, or ``None`` to disable. |
| `inference.model.rope_scaling` | `dict[str, Any] | str | None` | `None` | RoPE scaling configuration as a dict (e.g. ``{rope_type="yarn", factor=4.0, original_max_position_embeddings=32768}``). Forwarded as ``--rope-scaling``. |

<a id="rl-inference-model-vlm"></a>
##### `inference.model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `inference.model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `inference.model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="rl-inference-parallel"></a>
#### `inference.parallel`

Multi-node and multi-GPU parallelism (TP, DP, PP).

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.parallel.tp` | `int` | `1` | Tensor parallel size. Forwarded to vLLM as ``--tensor-parallel-size``. |
| `inference.parallel.dp` | `int` | `1` | _≥1._ Data parallel size. Forwarded to vLLM as ``--data-parallel-size``. |

<a id="rl-inference-weight-broadcast"></a>
#### `inference.weight_broadcast`

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.weight_broadcast.type` | `'nccl' | 'filesystem'` | `'filesystem'` | Weight broadcast transport. |

<a id="rl-inference-kv-cache-offload"></a>
#### `inference.kv_cache_offload`

CPU KV cache offload for inference workers. Standard inference uses vLLM's ``OffloadingConnector``. Disaggregated P/D deployments combine it with NIXL through ``MultiConnector`` in the SLURM launcher.

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.kv_cache_offload.cpu_bytes` | `int` | `1000000000` | _>0._ CPU bytes available for KV cache offloading per worker. |

<a id="rl-inference-slurm"></a>
#### `inference.slurm`

SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally.

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.slurm.job_name` | `str` | `'prime-rl'` | SLURM job name. |
| `inference.slurm.project_dir` | `Path` | `'.'` | Path to the project root, used to source .env, activate .venv, and run uv sync. |
| `inference.slurm.template_path` | `Path | None` | `None` | SLURM template file. If None, uses the bundled single-node or multi-node template. |
| `inference.slurm.partition` | `str` | `'cluster'` | SLURM partition (#SBATCH --partition). |
| `inference.slurm.nodelist` | `str | None` | `None` | Comma-separated list of specific nodes to run on (#SBATCH --nodelist). |
| `inference.slurm.exclude` | `str | None` | `None` | Comma-separated list of nodes to exclude (#SBATCH --exclude). |
| `inference.slurm.account` | `str | None` | `None` | SLURM account to charge (#SBATCH --account). |
| `inference.slurm.time` | `str | None` | `None` | Maximum wall time, e.g. '24:00:00' or '7-00:00:00' (#SBATCH --time). |
| `inference.slurm.pre_run_command` | `str | None` | `None` | Shell command to run on the head node after cd, .env sourcing, and venv activation. Useful for cleanup like ``sudo pkill -f vllm``; wrap with ``srun bash -c '...'`` to fan out to all nodes. |

<a id="rl-inference-experimental"></a>
#### `inference.experimental`

<a id="rl-inference-deployment"></a>
#### `inference.deployment`

Discriminated union — set `inference.deployment.type` to one of `single_node`, `multi_node`, `disaggregated` and provide the matching sub-fields.

<a id="rl-inference-deployment-single-node"></a>
##### `inference.deployment.type = "single_node"` (SingleNodeInferenceDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `inference.deployment.type` | `'single_node'` | `'single_node'` |  |

<a id="rl-inference-deployment-multi-node"></a>
##### `inference.deployment.type = "multi_node"` (MultiNodeInferenceDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `inference.deployment.type` | `'multi_node'` | `'multi_node'` |  |
| `inference.deployment.num_nodes` | `int` | `2` | _≥1._ Inference nodes. |
| `inference.deployment.router_port` | `int` | `8000` | Port for the vllm-router. |
| `inference.deployment.backend_port` | `int` | `8100` | Port for vLLM backend instances. |
| `inference.deployment.router_policy` | `str` | `'consistent_hash'` | vllm-router routing policy (e.g. ``consistent_hash``, ``round_robin``). |

<a id="rl-inference-deployment-disaggregated"></a>
##### `inference.deployment.type = "disaggregated"` (DisaggregatedInferenceDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `inference.deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `inference.deployment.type` | `'disaggregated'` | `'disaggregated'` |  |
| `inference.deployment.num_prefill_nodes` | `int` | `1` | _≥1._ Total prefill nodes. |
| `inference.deployment.num_decode_nodes` | `int` | `1` | _≥1._ Total decode nodes. |
| `inference.deployment.num_prefill_replicas` | `int` | `1` | _≥1._ Independent prefill vLLM instances. Must evenly divide ``num_prefill_nodes``. |
| `inference.deployment.num_decode_replicas` | `int` | `1` | _≥1._ Independent decode vLLM instances. Must evenly divide ``num_decode_nodes``. |
| `inference.deployment.router_port` | `int` | `8000` | Port for the vllm-router on each replica. |
| `inference.deployment.prefill_port` | `int` | `8100` | Port for prefill vLLM instances. |
| `inference.deployment.decode_port` | `int` | `8200` | Port for decode vLLM instances. |
| `inference.deployment.router_policy` | `str` | `'consistent_hash'` | vllm-router routing policy (e.g. ``consistent_hash``, ``round_robin``). |
| `inference.deployment.prefill_env_overrides` | `dict[str, str]` | `{}` | Extra environment variables exported only on prefill nodes. |
| `inference.deployment.decode_env_overrides` | `dict[str, str]` | `{}` | Extra environment variables exported only on decode nodes. |

<a id="rl-log"></a>
### `log`

Shared log config. Propagated to trainer and orchestrator.

| Field | Type | Default | Description |
|---|---|---|---|
| `log.level` | `str | None` | `None` | Log level for trainer and orchestrator. When unset, each sub-config's own log level applies (defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``). |
| `log.json_logging` | `bool` | `False` | Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.). |

<a id="rl-ckpt"></a>
### `ckpt`

Shared checkpoint config. If None, falls back to the sub-config checkpoint settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `ckpt.output_dir` | `Path | None` | `None` | Override directory for checkpoints and weights. When set, checkpoints and weight snapshots are written here instead of under the trainer ``output_dir``. |
| `ckpt.interval` | `int | None` | `None` | Interval at which to save checkpoints. |
| `ckpt.resume_step` | `int | None` | `None` | Step to resume from. If None, does not resume from a checkpoint. |
| `ckpt.keep_last` | `int | None` | `None` | _≥1._ Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency. |
| `ckpt.keep_interval` | `int | None` | `None` | _≥1._ Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping. |

<a id="rl-wandb"></a>
### `wandb`

Shared W&B config. If None, falls back to the sub-config W&B settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `wandb.project` | `str | None` | `'prime-rl'` | W&B project. |
| `wandb.entity` | `str | None` | `None` | W&B entity. |
| `wandb.name` | `str | None` | `None` | W&B run name. |
| `wandb.group` | `str | None` | `None` | W&B group. |
| `wandb.tags` | `list[str] | None` | `None` | W&B tags attached to the run. |
| `wandb.offline` | `bool | None` | `False` | Run W&B in offline mode. |
| `wandb.shared` | `bool` | `True` | Log trainer and orchestrator metrics to a single shared W&B run. Requires wandb SDK ≥ 0.19.9. Incompatible with offline mode. |

<a id="rl-model"></a>
### `model`

Shared model config. If None, falls back to the sub-config model settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |

<a id="rl-model-vlm"></a>
#### `model.vlm`

VLM configuration. Set this to enable vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="rl-tokenizer"></a>
### `tokenizer`

Shared tokenizer config. Propagated to trainer, orchestrator, and inference. If None, each component uses its own tokenizer config (defaulting to model name).

| Field | Type | Default | Description |
|---|---|---|---|
| `tokenizer.name` | `str | None` | `None` | Tokenizer name or path. If None, the model's default tokenizer is used. |
| `tokenizer.trust_remote_code` | `bool | None` | `None` | Trust remote code when initializing the tokenizer. If None, inherits the model's ``trust_remote_code`` setting. |
| `tokenizer.chat_template` | `str | None` | `None` | Chat template for the tokenizer. Either a Jinja2 template string or a path to a template file. If None, the tokenizer's default chat template is used. |

<a id="rl-weight-broadcast"></a>
### `weight_broadcast`

| Field | Type | Default | Description |
|---|---|---|---|
| `weight_broadcast.type` | `'nccl' | 'filesystem'` | `'filesystem'` | Weight broadcast transport. |
| `weight_broadcast.port` | `int` | `29501` | Port for NCCL weight broadcast. |
| `weight_broadcast.timeout` | `int` | `1200` | Timeout in seconds for NCCL weight broadcast. |
| `weight_broadcast.quantize_in_weight_transfer` | `bool` | `False` | Use kernel-format FP8 quantized NCCL transfer for weight updates. When disabled, uses default HF checkpoint-format transfer. |

<a id="rl-slurm"></a>
### `slurm`

SLURM configuration. If None, runs locally.

| Field | Type | Default | Description |
|---|---|---|---|
| `slurm.job_name` | `str` | `'prime-rl'` | SLURM job name. |
| `slurm.project_dir` | `Path` | `'.'` | Path to the project root, used to source .env, activate .venv, and run uv sync. |
| `slurm.template_path` | `Path | None` | `None` | SLURM template file. If None, uses the bundled single-node or multi-node template. |
| `slurm.partition` | `str` | `'cluster'` | SLURM partition (#SBATCH --partition). |
| `slurm.nodelist` | `str | None` | `None` | Comma-separated list of specific nodes to run on (#SBATCH --nodelist). |
| `slurm.exclude` | `str | None` | `None` | Comma-separated list of nodes to exclude (#SBATCH --exclude). |
| `slurm.account` | `str | None` | `None` | SLURM account to charge (#SBATCH --account). |
| `slurm.time` | `str | None` | `None` | Maximum wall time, e.g. '24:00:00' or '7-00:00:00' (#SBATCH --time). |
| `slurm.pre_run_command` | `str | None` | `None` | Shell command to run on the head node after cd, .env sourcing, and venv activation. Useful for cleanup like ``sudo pkill -f vllm``; wrap with ``srun bash -c '...'`` to fan out to all nodes. |

<a id="rl-experimental"></a>
### `experimental`

<a id="rl-deployment"></a>
### `deployment`

Discriminated union — set `deployment.type` to one of `single_node`, `multi_node` and provide the matching sub-fields.

<a id="rl-deployment-single-node"></a>
#### `deployment.type = "single_node"` (SingleNodeDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `deployment.type` | `'single_node'` | `'single_node'` |  |
| `deployment.num_train_gpus` | `int` | `1` | GPUs allocated to the trainer. |
| `deployment.num_infer_gpus` | `int` | `1` | GPUs allocated to inference. |

<a id="rl-deployment-multi-node"></a>
#### `deployment.type = "multi_node"` (MultiNodeDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `deployment.type` | `'multi_node'` | `'multi_node'` |  |
| `deployment.num_train_nodes` | `int` | *required* | Training nodes. |
| `deployment.num_infer_nodes` | `int` | *required* | _≥0._ Inference nodes per replica. Set to 0 to skip inference and orchestrator (requires fake data). |
| `deployment.num_infer_replicas` | `int` | `1` | _≥1._ Independent inference replicas. Total inference nodes = ``num_infer_nodes * num_infer_replicas``. |
| `deployment.nodes_per_fsdp_group` | `int | None` | `None` | Training nodes per FSDP island. Auto-sets ``trainer.dp_replicate = num_train_nodes / nodes_per_fsdp_group``. |

<a id="sft"></a>
## `sft` — Supervised fine-tuning

The `sft` entrypoint runs supervised fine-tuning on a tokenized dataset.

_Defined in_ `prime_rl.configs.sft.SFTConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `use_renderer` | `bool` | `False` | Tokenize SFT samples through the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) library (single ``render()`` + ``message_indices`` mask) instead of the default ``build_incremental_token_mask`` path. Required for chat templates that render position-dependently (e.g. Qwen3, Qwen3.5). |
| `output_dir` | `Path` | `'outputs'` | Directory to write outputs to — checkpoints and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node. |
| `clean_output_dir` | `bool` | `False` | Delete the output directory before starting training. Required to overwrite an output directory that contains checkpoints from a previous run when not resuming. |
| `matmul_precision` | `'highest' | 'high' | 'medium'` | `'high'` | Precision for float32 matrix multiplications. ``highest`` is full FP32 (required on ROCm/AMD GPUs to avoid catastrophic precision loss in softmax over large vocabularies). ``high`` enables TF32 on NVIDIA GPUs for a speedup with minor precision tradeoff. See ``torch.set_float32_matmul_precision``. |
| `max_steps` | `int | None` | `None` | Maximum training steps. If None, runs indefinitely. |
| `memory_profiler_path` | `Path | None` | `None` | Path to write the memory profile to. |
| `trace_path` | `Path | None` | `None` | Path to write the PyTorch profiler trace to. |
| `dist_timeout_seconds` | `int` | `600` | Timeout in seconds for torch distributed ops. |
| `loss_impl` | `'liger' | 'torch' | 'liger_fused' | 'quack_fused'` | `'torch'` | Cross-entropy loss implementation. ``liger_fused`` fuses the lm_head projection with the CE loss to avoid materializing full logits. ``quack_fused`` uses quack-kernels for chunked linear + CE with CuTe DSL CUDA kernels. |
| `dry_run` | `bool` | `False` | Only validate and dump resolved configs, then exit early. |

<a id="sft-model"></a>
### `model`

| Field | Type | Default | Description |
|---|---|---|---|
| `model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `model.trust_remote_code` | `bool` | `False` | Trust remote code when initializing the tokenizer. |
| `model.seq_len` | `int` | `2048` | Sequence length the model is trained on. |
| `model.attn` | `'eager' | 'sdpa' | 'flash_attention_2' | 'flash_attention_3' | 'fa4'` | `'flash_attention_2'` | Attention implementation. With CP enabled, ring attention uses the matching kernel family (FA2/FA3/FA4). |
| `model.fsdp_cpu_offload` | `bool` | `False` | Enable FSDP CPU offloading for parameters, gradients, and optimizer states. Uses pinned memory for efficient CPU↔GPU transfers. |
| `model.optim_cpu_offload` | `bool` | `False` | Offload only optimizer states (momentum, variance) to CPU, keeping weights on GPU. Avoids the H2D all-gather overhead of FSDP CPU offload while still saving GPU memory. |
| `model.reshard_after_forward` | `bool` | `True` | Reshard the model after each forward pass. |
| `model.dp_replicate` | `int` | `1` | Data parallel dim where model weights are replicated. |
| `model.ep` | `int` | `1` | Expert parallelism degree for MoE layers. 1 disables EP. |
| `model.ep_comm_backend` | `'torch' | 'deepep'` | `'torch'` | Communication backend for expert parallelism. ``torch`` uses TorchTitan all-to-all collectives; ``deepep`` uses DeepEP custom kernels. |
| `model.deepep_num_sms` | `int` | `20` | _≥1._ SMs allocated for DeepEP intranode dispatch/combine kernels. Also determines internode RDMA channel count (``num_channels = num_sms / 2``). Lower values leave more SMs for compute; higher values speed up dispatch/combine. The optimal value depends on EP degree and hardware. Only used when ``ep_comm_backend='deepep'``. |
| `model.deepep_token_chunk_size` | `int | None` | `None` | _≥1._ Token chunk size for DeepEP MoE pipelining. When set, DeepEP dispatch for chunk i+1 is launched while experts compute chunk i. Only used when ``ep_comm_backend='deepep'``. |
| `model.cp` | `int` | `1` | Context parallelism degree. 1 disables CP. |
| `model.cp_style` | `'ring' | 'ulysses'` | `'ring'` | CP communication style. ``ring`` uses ring-attention all-gather/reduce-scatter (requires custom kernels per attention type). ``ulysses`` uses all-to-all to redistribute Q/K/V from sequence-sharded to head-sharded, runs vanilla attention locally on the full sequence, then all-to-all back — works out-of-the-box with any attention kernel (softmax FA, linear attention, mamba, etc.). |
| `model.impl` | `'hf' | 'custom' | 'auto'` | `'auto'` | Model implementation. ``auto`` selects ``custom`` if supported by the model, otherwise ``hf``. |
| `model.optimization_dtype` | `'bfloat16' | 'float32'` | `'float32'` | dtype for model optimization. |
| `model.reduce_dtype` | `'bfloat16' | 'float32'` | `'float32'` | dtype for gradient/parameter reductions. |
| `model.moe_use_grouped_mm` | `bool` | `True` | Use grouped mm for MoE layers. Requires compute capability ≥ 9.0. |
| `model.fp8` | `bool` | `False` | FP8 training via DeepGEMM. Replaces ``nn.Linear`` with FP8 blockwise linear and uses FP8 grouped GEMM for MoE experts. Requires SM90 (Hopper) GPUs and ``model.impl='custom'``. |
| `model.freeze_moe_router` | `bool` | `False` | Freeze MoE router parameters during training. |
| `model.fused_lm_head_token_chunk_size` | `int | 'auto' | 'disabled'` | `'disabled'` | Flattened token chunk size for the fused LM head. ``int >= 1`` sets the tokens per LM-head chunk explicitly; ``auto`` auto-enables (RL training picks 8192); ``disabled`` uses the vanilla LM head. Integer values aren't supported for SFT training. |

<a id="sft-model-vlm"></a>
#### `model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="sft-model-compile"></a>
#### `model.compile`

Compile the model with ``torch.compile``.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.compile.fullgraph` | `bool` | `False` | Compile transformer blocks with ``fullgraph=True``. |

<a id="sft-model-ac"></a>
#### `model.ac`

Activation checkpointing configuration. If None, activation checkpointing is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.ac.mode` | `'full' | 'selective'` | `'full'` | ``full`` checkpoints whole transformer blocks; ``selective`` checkpoints only the subcomponents listed in ``targets`` inside supported custom decoder layers. |
| `model.ac.freq` | `int` | `1` | _≥1._ Apply activation checkpointing to every N layers. |
| `model.ac.targets` | `list[str]` | `['norm']` | Selective checkpoint targets. ``norm`` checkpoints every norm module inside selected layers. ``attn_proj`` checkpoints projection-side attention work outside the kernel (input/output projections, attention-local norms, RoPE, gating, model-specific MLA projection helpers). ``mlp`` checkpoints the entire dense MLP forward (not for MoE). ``mla_up_proj`` checkpoints MLA Q/KV up-projection where supported. ``routed_experts`` checkpoints routed expert compute in MoE layers (including LatentMoE). ``linear_attn`` checkpoints non-softmax token mixers (NemotronH Mamba, Qwen3.5-MoE GatedDeltaNet, AFMoE sliding-window attention). |

<a id="sft-model-ac-offloading"></a>
#### `model.ac_offloading`

Activation offloading configuration. If None, activation offloading is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.ac_offloading.pin_memory` | `bool` | `True` | Pin offloaded activations to CPU memory. |
| `model.ac_offloading.max_inflight_activations` | `int` | `5` | _≥1._ Max activations kept in flight while offloading. More activations smooth overlap at the cost of GPU memory. |

<a id="sft-model-index-cache"></a>
#### `model.index_cache`

DSA IndexCache sub-configuration. If set, sparse-attention top-k indices are reused across decoder layers per the configured schedule (mirrors vLLM's IndexCache HF overrides). If None, every layer recomputes its own indices.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.index_cache.topk_freq` | `int` | `1` | _≥1._ Recompute DSA top-k indices every N layers; intervening layers reuse the cached indices. ``1`` recomputes every layer (effectively no reuse). Mirrors vLLM's ``index_topk_freq`` HF override. |
| `model.index_cache.topk_pattern` | `str | None` | `None` | Optional per-layer schedule that overrides ``topk_freq``. ``'F'`` computes fresh indices for that layer; ``'S'`` reuses the previously cached indices. Length should match the number of decoder layers. |

<a id="sft-model-lora"></a>
#### `model.lora`

LoRA configuration. If None, LoRA is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.lora.rank` | `int` | `16` | _≥1._ Rank of the low-rank decomposition matrices. |
| `model.lora.alpha` | `float` | `32.0` | _≥0._ LoRA scaling parameter. |
| `model.lora.dropout` | `float` | `0.0` | _≥0, ≤1._ LoRA dropout rate. |
| `model.lora.target_modules` | `list[str]` | `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'experts', 'fc1_latent_proj', 'fc2_latent_proj']` | Module names or regex patterns to apply LoRA to. Simple names (e.g. ``q_proj``) match any component in the module path; regex patterns match anywhere in the name. Names unknown to the current model are silently ignored, so defaults cover multiple architectures. NemotronH note: ``experts`` matches NonGatedGroupedExperts inside LatentMoE; ``fc1_latent_proj``/``fc2_latent_proj`` adapt the latent up/down projections. Add ``in_proj``/``out_proj`` to also LoRA Mamba. |
| `model.lora.modules_to_save` | `list[str]` | `[]` | Module names or regex patterns to keep fully trainable (not freeze). Same matching rules as ``target_modules``. |

<a id="sft-model-debug"></a>
#### `model.debug`

Debugging knobs for the model and distributed training.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.debug.num_layers` | `int | None` | `None` | Override the number of transformer layers (truncates the model). |
| `model.debug.random_init` | `bool` | `False` | Randomly initialize the model instead of loading weights. |
| `model.debug.force_balanced_routing` | `bool` | `False` | Replace MoE token-choice routing with a round-robin assignment so every expert sees an equal share. Intended for fake-data smoke tests where untrained routing would otherwise OOM under severe imbalance. Gating scores are still gathered from the override indices so the forward pass stays consistent. |

<a id="sft-tokenizer"></a>
### `tokenizer`

| Field | Type | Default | Description |
|---|---|---|---|
| `tokenizer.name` | `str | None` | `None` | Tokenizer name or path. If None, the model's default tokenizer is used. |
| `tokenizer.trust_remote_code` | `bool | None` | `None` | Trust remote code when initializing the tokenizer. If None, inherits the model's ``trust_remote_code`` setting. |
| `tokenizer.chat_template` | `str | None` | `None` | Chat template for the tokenizer. Either a Jinja2 template string or a path to a template file. If None, the tokenizer's default chat template is used. |

<a id="sft-renderer"></a>
### `renderer`

Client-side renderer configuration. Only consumed when ``use_renderer=true``.

| Field | Type | Default | Description |
|---|---|---|---|
| `renderer.name` | `str` | `'auto'` | Renderer used for chat-template tokenization. One of: ``auto`` (detect from tokenizer), ``qwen3``, ``qwen3_vl``, ``qwen3.5``, ``glm5``, ``glm4.5``, ``minimax-m2``, ``deepseek_v3``, ``kimi_k2``, ``kimi_k25``, ``nemotron3``, ``gpt_oss``, ``default``. |
| `renderer.tool_parser` | `str | None` | `None` | Tool parser from [`renderers.parsers`](https://github.com/PrimeIntellect-ai/renderers). Only consumed by DefaultRenderer; model-specific renderers bake their own parsing in. Options: ``qwen3``, ``qwen3.5``, ``glm``, ``deepseek_v3``. |
| `renderer.reasoning_parser` | `str | None` | `None` | Reasoning parser from [`renderers.parsers`](https://github.com/PrimeIntellect-ai/renderers). Only consumed by DefaultRenderer. Options: ``think``. |
| `renderer.pool_size` | `int | None` | `None` | _≥1._ Number of renderer slots shared across concurrent rollouts. Bump for long multi-turn prompts where client-side jinja tokenization serializes. |
| `renderer.preserve_all_thinking` | `bool` | `False` | Re-emit every past-assistant turn's ``reasoning_content`` between ``<think>``/``</think>`` (or the model's equivalent), even when the chat template would drop it. Strict superset of preserve_thinking_between_tool_calls. |
| `renderer.preserve_thinking_between_tool_calls` | `bool` | `False` | Preserve past-assistant ``reasoning_content`` only inside the current tool cycle — the contiguous assistant→tool→…→assistant block after the most recent user message, when that block contains at least one tool response. A new user turn closes the block. |

<a id="sft-val"></a>
### `val`

Validation configuration. If None, no validation runs.

| Field | Type | Default | Description |
|---|---|---|---|
| `val.interval` | `int` | `50` | _≥1._ Run validation every N training steps. |
| `val.eval_on_start` | `bool` | `False` | Run validation before the first training step. |

<a id="sft-val-data"></a>
#### `val.data`

| Field | Type | Default | Description |
|---|---|---|---|
| `val.data.batch_size` | `int` | `128` | _≥1._ Global batch size. |
| `val.data.seq_len` | `int` | `128` | _≥1._ Sequence length. |
| `val.data.pack_function` | `'cat' | 'stack'` | `'cat'` | Sample packing strategy. ``cat`` concatenates; ``stack`` requires ``seq_len`` divisible by 256. |
| `val.data.micro_batch_size` | `int` | `1` | _≥1._ Per-step micro batch size. ``batch_size`` must be divisible by this. |
| `val.data.type` | `'sft'` | `'sft'` |  |
| `val.data.name` | `str` | `'PrimeIntellect/Reverse-Text-SFT'` | HF dataset name or path. |
| `val.data.subsets` | `list[str] | None` | `None` | Subsets to load from the HF dataset. |
| `val.data.splits` | `list[str] | None` | `None` | Splits to load from the HF dataset. |
| `val.data.probabilities` | `list[float] | None` | `None` | Sampling probabilities for each subset/split. |
| `val.data.stopping_strategy` | `'first_exhausted' | 'all_exhausted'` | `'all_exhausted'` | Stopping strategy when interleaving multiple subsets/splits. |
| `val.data.shuffle` | `bool` | `True` | Shuffle the dataset at the start of each epoch. |
| `val.data.seed` | `int` | `0` | Random seed for shuffling. Re-shuffled per epoch by adding the epoch count to the seed. |

<a id="sft-val-data-loss-mask"></a>
##### `val.data.loss_mask`

Which message types contribute to the loss.

| Field | Type | Default | Description |
|---|---|---|---|
| `val.data.loss_mask.system` | `bool` | `False` | System messages contribute to the loss. |
| `val.data.loss_mask.user` | `bool` | `False` | User messages contribute to the loss. |
| `val.data.loss_mask.assistant` | `bool` | `True` | Assistant messages contribute to the loss. |
| `val.data.loss_mask.tool` | `bool` | `False` | Tool messages contribute to the loss. |

<a id="sft-ckpt"></a>
### `ckpt`

| Field | Type | Default | Description |
|---|---|---|---|
| `ckpt.output_dir` | `Path | None` | `None` | Override directory for checkpoints and weights. If set, checkpoints and weight snapshots are written here instead of under the trainer ``output_dir`` — useful for writing large checkpoints to a separate storage volume. |
| `ckpt.interval` | `int | None` | `None` | _≥1._ Interval at which to save the training checkpoint. If None, only checkpoints at the end of training. |
| `ckpt.skip_gather_master_weights` | `bool` | `False` | Skip gathering and saving HF-compatible weight checkpoints. Useful for large models where the gather is expensive and only DCP checkpoints are needed. |
| `ckpt.weights_only` | `bool` | `False` | Save only weight checkpoints (no optimizer/scheduler state). Much faster and smaller than full checkpoints, but cannot resume training. |
| `ckpt.resume_step` | `int | None` | `None` | _≥-1._ Step to resume training from. None starts from scratch; ``-1`` restarts from the latest checkpoint available. |
| `ckpt.keep_last` | `int | None` | `None` | _≥1._ Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency. |
| `ckpt.keep_interval` | `int | None` | `None` | _≥1._ Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping. |
| `ckpt.skip_progress` | `bool` | `False` | Skip loading the progress from checkpoint. |
| `ckpt.skip_scheduler` | `bool` | `False` | Skip loading the scheduler from checkpoint. |
| `ckpt.skip_dataloader` | `bool` | `False` | Skip loading the dataloader from checkpoint. |
| `ckpt.skip_optimizer` | `bool` | `False` | Skip loading the optimizer state from checkpoint. |

<a id="sft-ckpt-weights"></a>
#### `ckpt.weights`

Weight-checkpoint sub-configuration. If None, no HF-compatible weight checkpoints are written.

| Field | Type | Default | Description |
|---|---|---|---|
| `ckpt.weights.save_sharded` | `bool` | `True` | Save the weight checkpoint in sharded format. |
| `ckpt.weights.save_format` | `'safetensors' | 'torch'` | `'safetensors'` | Weight checkpoint serialization format. |
| `ckpt.weights.save_adapter_separately` | `bool` | `False` | Save LoRA adapters separately before merging into full model weights. |

<a id="sft-log"></a>
### `log`

| Field | Type | Default | Description |
|---|---|---|---|
| `log.level` | `str` | `'info'` | Log level for the process. Defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``. |
| `log.vf_level` | `str` | `'info'` | Log level for the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) package. Defaults to ``$PRIME_VF_LOG_LEVEL`` if set, else ``info``. |
| `log.json_logging` | `bool` | `False` | Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.). |
| `log.log_data` | `bool` | `False` | Log the first data sample at startup. |
| `log.ranks_filter` | `list[int]` | `[0]` | Trainer ranks to show in console output. Passed to ``torchrun --local-ranks-filter``. |

<a id="sft-wandb"></a>
### `wandb`

| Field | Type | Default | Description |
|---|---|---|---|
| `wandb.project` | `str` | `'prime-rl'` | W&B project to log to. |
| `wandb.entity` | `str | None` | `None` | W&B entity to log to. |
| `wandb.name` | `str | None` | `None` | W&B run name. |
| `wandb.group` | `str | None` | `None` | W&B group. |
| `wandb.tags` | `list[str] | None` | `None` | W&B tags attached to the run. |
| `wandb.offline` | `bool` | `False` | Run W&B in offline mode. |

<a id="sft-bench"></a>
### `bench`

Benchmark-mode configuration. When set, ``max_steps`` is forced to 4 and fake data is used.

| Field | Type | Default | Description |
|---|---|---|---|
| `bench.output_json` | `Path | None` | `None` | Path to write benchmark results as JSON. If unset, results are only printed to the console. |

<a id="sft-gc"></a>
### `gc`

Garbage collection config. Disables automatic GC and runs deterministic collections every N steps to avoid stragglers. Set to null to use Python's default GC behavior.

| Field | Type | Default | Description |
|---|---|---|---|
| `gc.interval` | `int` | `50` | _≥1._ Run garbage collection every N training steps. Disables Python's automatic GC so every rank collects together and one slow rank can't stall the others. |

<a id="sft-heartbeat"></a>
### `heartbeat`

BetterStack heartbeat configuration for monitoring training progress.

| Field | Type | Default | Description |
|---|---|---|---|
| `heartbeat.url` | `str` | *required* | URL to send the heartbeat to. |

<a id="sft-slurm"></a>
### `slurm`

SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally.

| Field | Type | Default | Description |
|---|---|---|---|
| `slurm.job_name` | `str` | `'prime-rl'` | SLURM job name. |
| `slurm.project_dir` | `Path` | `'.'` | Path to the project root, used to source .env, activate .venv, and run uv sync. |
| `slurm.template_path` | `Path | None` | `None` | SLURM template file. If None, uses the bundled single-node or multi-node template. |
| `slurm.partition` | `str` | `'cluster'` | SLURM partition (#SBATCH --partition). |
| `slurm.nodelist` | `str | None` | `None` | Comma-separated list of specific nodes to run on (#SBATCH --nodelist). |
| `slurm.exclude` | `str | None` | `None` | Comma-separated list of nodes to exclude (#SBATCH --exclude). |
| `slurm.account` | `str | None` | `None` | SLURM account to charge (#SBATCH --account). |
| `slurm.time` | `str | None` | `None` | Maximum wall time, e.g. '24:00:00' or '7-00:00:00' (#SBATCH --time). |
| `slurm.pre_run_command` | `str | None` | `None` | Shell command to run on the head node after cd, .env sourcing, and venv activation. Useful for cleanup like ``sudo pkill -f vllm``; wrap with ``srun bash -c '...'`` to fan out to all nodes. |

<a id="sft-experimental"></a>
### `experimental`

<a id="sft-data"></a>
### `data`

Discriminated union — set `data.type` to one of `fake`, `sft` and provide the matching sub-fields.

<a id="sft-data-fake"></a>
#### `data.type = "fake"` (FakeDataConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `data.batch_size` | `int` | `128` | _≥1._ Global batch size. |
| `data.seq_len` | `int` | `128` | _≥1._ Sequence length. |
| `data.pack_function` | `'cat' | 'stack'` | `'cat'` | Sample packing strategy. ``cat`` concatenates; ``stack`` requires ``seq_len`` divisible by 256. |
| `data.micro_batch_size` | `int` | `1` | _≥1._ Per-step micro batch size. ``batch_size`` must be divisible by this. |
| `data.type` | `'fake'` | `'fake'` |  |
| `data.length` | `'fixed' | 'variable'` | `'fixed'` | Use fixed-length samples or variable-length samples. |
| `data.input_ids` | `'increasing' | 'random'` | `'increasing'` | Token id generator: ``increasing`` for deterministic sequences, ``random`` for random ids. |

<a id="sft-data-sft"></a>
#### `data.type = "sft"` (SFTDataConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `data.batch_size` | `int` | `128` | _≥1._ Global batch size. |
| `data.seq_len` | `int` | `128` | _≥1._ Sequence length. |
| `data.pack_function` | `'cat' | 'stack'` | `'cat'` | Sample packing strategy. ``cat`` concatenates; ``stack`` requires ``seq_len`` divisible by 256. |
| `data.micro_batch_size` | `int` | `1` | _≥1._ Per-step micro batch size. ``batch_size`` must be divisible by this. |
| `data.type` | `'sft'` | `'sft'` |  |
| `data.name` | `str` | `'PrimeIntellect/Reverse-Text-SFT'` | HF dataset name or path. |
| `data.subsets` | `list[str] | None` | `None` | Subsets to load from the HF dataset. |
| `data.splits` | `list[str] | None` | `None` | Splits to load from the HF dataset. |
| `data.probabilities` | `list[float] | None` | `None` | Sampling probabilities for each subset/split. |
| `data.stopping_strategy` | `'first_exhausted' | 'all_exhausted'` | `'all_exhausted'` | Stopping strategy when interleaving multiple subsets/splits. |
| `data.shuffle` | `bool` | `True` | Shuffle the dataset at the start of each epoch. |
| `data.seed` | `int` | `0` | Random seed for shuffling. Re-shuffled per epoch by adding the epoch count to the seed. |

<a id="sft-data-sft-loss-mask"></a>
##### `data.loss_mask`

Which message types contribute to the loss.

| Field | Type | Default | Description |
|---|---|---|---|
| `data.loss_mask.system` | `bool` | `False` | System messages contribute to the loss. |
| `data.loss_mask.user` | `bool` | `False` | User messages contribute to the loss. |
| `data.loss_mask.assistant` | `bool` | `True` | Assistant messages contribute to the loss. |
| `data.loss_mask.tool` | `bool` | `False` | Tool messages contribute to the loss. |

<a id="sft-optim"></a>
### `optim`

Discriminated union — set `optim.type` to one of `sgd`, `adamw`, `muon`, `sign_sgd` and provide the matching sub-fields.

<a id="sft-optim-sgd"></a>
#### `optim.type = "sgd"` (SGDConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `optim.type` | `'sgd'` | `'sgd'` |  |
| `optim.nesterov` | `bool` | `True` | Use Nesterov momentum. |
| `optim.momentum` | `float` | `0.9` | SGD momentum factor. |

<a id="sft-optim-adamw"></a>
#### `optim.type = "adamw"` (AdamWConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `optim.type` | `'adamw'` | `'adamw'` |  |
| `optim.betas1` | `float` | `0.9` | _≥0._ Adam first-moment (β1) decay. |
| `optim.betas2` | `float` | `0.999` | _≥0._ Adam second-moment (β2) decay. |

<a id="sft-optim-muon"></a>
#### `optim.type = "muon"` (MuonConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `optim.type` | `'muon'` | `'muon'` |  |
| `optim.mu` | `float` | `0.95` | _≥0._ Momentum factor for the Muon algorithm. |
| `optim.betas1` | `float` | `0.9` | _≥0._ β1 for the AdamW/Lion sub-optimizer used on non-Muon params. |
| `optim.betas2` | `float` | `0.95` | _≥0._ β2 for the AdamW/Lion sub-optimizer used on non-Muon params. |

<a id="sft-optim-sign-sgd"></a>
#### `optim.type = "sign_sgd"` (SignSGDConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `optim.type` | `'sign_sgd'` | `'sign_sgd'` |  |

<a id="sft-scheduler"></a>
### `scheduler`

Discriminated union — set `scheduler.type` to one of `constant`, `linear`, `cosine` and provide the matching sub-fields.

<a id="sft-scheduler-constant"></a>
#### `scheduler.type = "constant"` (ConstantSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `scheduler.type` | `'constant'` | `'constant'` |  |

<a id="sft-scheduler-linear"></a>
#### `scheduler.type = "linear"` (LinearSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `scheduler.type` | `'linear'` | `'linear'` |  |
| `scheduler.warmup_steps` | `int` | `10` | _≥0._ Warmup steps for the learning rate scheduler. |
| `scheduler.decay_steps` | `int` | `10` | _≥0._ Steps to decay the learning rate during the final portion of training. |
| `scheduler.min_lr` | `float` | `0.0` | _≥0._ Minimum learning rate to converge to. |

<a id="sft-scheduler-cosine"></a>
#### `scheduler.type = "cosine"` (CosineSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `scheduler.type` | `'cosine'` | `'cosine'` |  |
| `scheduler.warmup_steps` | `int` | `10` | _≥0._ Warmup steps for the learning rate scheduler. |
| `scheduler.min_lr` | `float` | `0.0` | _≥0._ Minimum learning rate to converge to. |

<a id="sft-deployment"></a>
### `deployment`

Discriminated union — set `deployment.type` to one of `single_node`, `multi_node` and provide the matching sub-fields.

<a id="sft-deployment-single-node"></a>
#### `deployment.type = "single_node"` (SingleNodeDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `deployment.type` | `'single_node'` | `'single_node'` |  |
| `deployment.num_gpus` | `int` | `1` | GPUs to use. |

<a id="sft-deployment-multi-node"></a>
#### `deployment.type = "multi_node"` (MultiNodeDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `deployment.type` | `'multi_node'` | `'multi_node'` |  |
| `deployment.num_nodes` | `int` | `2` | Training nodes. |
| `deployment.nodes_per_fsdp_group` | `int | None` | `None` | Nodes per FSDP island. Auto-sets ``model.dp_replicate = num_nodes / nodes_per_fsdp_group``. |

<a id="trainer"></a>
## `trainer` — Standalone trainer

The `trainer` entrypoint runs only the trainer process. It expects rollouts to be shipped in via the configured transport (filesystem or ZMQ) by an external orchestrator.

_Defined in_ `prime_rl.configs.trainer.TrainerConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `Path` | `'outputs'` | Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node. |
| `matmul_precision` | `'highest' | 'high' | 'medium'` | `'high'` | Precision for float32 matrix multiplications. ``highest`` is full FP32 (required on ROCm/AMD GPUs to avoid catastrophic precision loss in softmax over large vocabularies). ``high`` enables TF32 on NVIDIA GPUs for a speedup with minor precision tradeoff. See ``torch.set_float32_matmul_precision``. |
| `max_steps` | `int | None` | `None` | Maximum number of training steps. If None, runs indefinitely. |
| `max_async_level` | `int` | `1` | _≥0._ Maximum steps inference can be ahead of training (how off-policy inference can be). Higher values yield better throughput via async execution at the cost of policy lag; ``0`` is fully synchronous. |
| `enable_router_replay` | `bool` | `False` | Return routed experts in the batch so the trainer can replay routing. Requires ``enable_return_routed_experts=true`` on the vLLM server (or ``--enable-return-routed-experts``) and is only supported for custom models. |
| `memory_profiler_path` | `Path | None` | `None` | Path to write the memory profile to. |
| `trace_path` | `Path | None` | `None` | Path to write the PyTorch profiler trace to. |
| `dist_timeout_seconds` | `int` | `600` | Timeout in seconds for torch distributed ops. |
| `max_concurrent_runs` | `int` | `1` | _≥1._ Maximum number of concurrent runs to allow. If 1, only one run may run at a time. |

<a id="trainer-model"></a>
### `model`

| Field | Type | Default | Description |
|---|---|---|---|
| `model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `model.trust_remote_code` | `bool` | `False` | Trust remote code when initializing the tokenizer. |
| `model.seq_len` | `int` | `2048` | Sequence length the model is trained on. |
| `model.attn` | `'eager' | 'sdpa' | 'flash_attention_2' | 'flash_attention_3' | 'fa4'` | `'flash_attention_2'` | Attention implementation. With CP enabled, ring attention uses the matching kernel family (FA2/FA3/FA4). |
| `model.fsdp_cpu_offload` | `bool` | `False` | Enable FSDP CPU offloading for parameters, gradients, and optimizer states. Uses pinned memory for efficient CPU↔GPU transfers. |
| `model.optim_cpu_offload` | `bool` | `False` | Offload only optimizer states (momentum, variance) to CPU, keeping weights on GPU. Avoids the H2D all-gather overhead of FSDP CPU offload while still saving GPU memory. |
| `model.reshard_after_forward` | `bool` | `True` | Reshard the model after each forward pass. |
| `model.dp_replicate` | `int` | `1` | Data parallel dim where model weights are replicated. |
| `model.ep` | `int` | `1` | Expert parallelism degree for MoE layers. 1 disables EP. |
| `model.ep_comm_backend` | `'torch' | 'deepep'` | `'torch'` | Communication backend for expert parallelism. ``torch`` uses TorchTitan all-to-all collectives; ``deepep`` uses DeepEP custom kernels. |
| `model.deepep_num_sms` | `int` | `20` | _≥1._ SMs allocated for DeepEP intranode dispatch/combine kernels. Also determines internode RDMA channel count (``num_channels = num_sms / 2``). Lower values leave more SMs for compute; higher values speed up dispatch/combine. The optimal value depends on EP degree and hardware. Only used when ``ep_comm_backend='deepep'``. |
| `model.deepep_token_chunk_size` | `int | None` | `None` | _≥1._ Token chunk size for DeepEP MoE pipelining. When set, DeepEP dispatch for chunk i+1 is launched while experts compute chunk i. Only used when ``ep_comm_backend='deepep'``. |
| `model.cp` | `int` | `1` | Context parallelism degree. 1 disables CP. |
| `model.cp_style` | `'ring' | 'ulysses'` | `'ring'` | CP communication style. ``ring`` uses ring-attention all-gather/reduce-scatter (requires custom kernels per attention type). ``ulysses`` uses all-to-all to redistribute Q/K/V from sequence-sharded to head-sharded, runs vanilla attention locally on the full sequence, then all-to-all back — works out-of-the-box with any attention kernel (softmax FA, linear attention, mamba, etc.). |
| `model.impl` | `'hf' | 'custom' | 'auto'` | `'auto'` | Model implementation. ``auto`` selects ``custom`` if supported by the model, otherwise ``hf``. |
| `model.optimization_dtype` | `'bfloat16' | 'float32'` | `'float32'` | dtype for model optimization. |
| `model.reduce_dtype` | `'bfloat16' | 'float32'` | `'float32'` | dtype for gradient/parameter reductions. |
| `model.moe_use_grouped_mm` | `bool` | `True` | Use grouped mm for MoE layers. Requires compute capability ≥ 9.0. |
| `model.fp8` | `bool` | `False` | FP8 training via DeepGEMM. Replaces ``nn.Linear`` with FP8 blockwise linear and uses FP8 grouped GEMM for MoE experts. Requires SM90 (Hopper) GPUs and ``model.impl='custom'``. |
| `model.freeze_moe_router` | `bool` | `False` | Freeze MoE router parameters during training. |
| `model.fused_lm_head_token_chunk_size` | `int | 'auto' | 'disabled'` | `'disabled'` | Flattened token chunk size for the fused LM head. ``int >= 1`` sets the tokens per LM-head chunk explicitly; ``auto`` auto-enables (RL training picks 8192); ``disabled`` uses the vanilla LM head. Integer values aren't supported for SFT training. |

<a id="trainer-model-vlm"></a>
#### `model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="trainer-model-compile"></a>
#### `model.compile`

Compile the model with ``torch.compile``.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.compile.fullgraph` | `bool` | `False` | Compile transformer blocks with ``fullgraph=True``. |

<a id="trainer-model-ac"></a>
#### `model.ac`

Activation checkpointing configuration. If None, activation checkpointing is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.ac.mode` | `'full' | 'selective'` | `'full'` | ``full`` checkpoints whole transformer blocks; ``selective`` checkpoints only the subcomponents listed in ``targets`` inside supported custom decoder layers. |
| `model.ac.freq` | `int` | `1` | _≥1._ Apply activation checkpointing to every N layers. |
| `model.ac.targets` | `list[str]` | `['norm']` | Selective checkpoint targets. ``norm`` checkpoints every norm module inside selected layers. ``attn_proj`` checkpoints projection-side attention work outside the kernel (input/output projections, attention-local norms, RoPE, gating, model-specific MLA projection helpers). ``mlp`` checkpoints the entire dense MLP forward (not for MoE). ``mla_up_proj`` checkpoints MLA Q/KV up-projection where supported. ``routed_experts`` checkpoints routed expert compute in MoE layers (including LatentMoE). ``linear_attn`` checkpoints non-softmax token mixers (NemotronH Mamba, Qwen3.5-MoE GatedDeltaNet, AFMoE sliding-window attention). |

<a id="trainer-model-ac-offloading"></a>
#### `model.ac_offloading`

Activation offloading configuration. If None, activation offloading is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.ac_offloading.pin_memory` | `bool` | `True` | Pin offloaded activations to CPU memory. |
| `model.ac_offloading.max_inflight_activations` | `int` | `5` | _≥1._ Max activations kept in flight while offloading. More activations smooth overlap at the cost of GPU memory. |

<a id="trainer-model-index-cache"></a>
#### `model.index_cache`

DSA IndexCache sub-configuration. If set, sparse-attention top-k indices are reused across decoder layers per the configured schedule (mirrors vLLM's IndexCache HF overrides). If None, every layer recomputes its own indices.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.index_cache.topk_freq` | `int` | `1` | _≥1._ Recompute DSA top-k indices every N layers; intervening layers reuse the cached indices. ``1`` recomputes every layer (effectively no reuse). Mirrors vLLM's ``index_topk_freq`` HF override. |
| `model.index_cache.topk_pattern` | `str | None` | `None` | Optional per-layer schedule that overrides ``topk_freq``. ``'F'`` computes fresh indices for that layer; ``'S'`` reuses the previously cached indices. Length should match the number of decoder layers. |

<a id="trainer-model-lora"></a>
#### `model.lora`

LoRA configuration. If None, LoRA is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.lora.rank` | `int` | `16` | _≥1._ Rank of the low-rank decomposition matrices. |
| `model.lora.alpha` | `float` | `32.0` | _≥0._ LoRA scaling parameter. |
| `model.lora.dropout` | `float` | `0.0` | _≥0, ≤1._ LoRA dropout rate. |
| `model.lora.target_modules` | `list[str]` | `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'experts', 'fc1_latent_proj', 'fc2_latent_proj']` | Module names or regex patterns to apply LoRA to. Simple names (e.g. ``q_proj``) match any component in the module path; regex patterns match anywhere in the name. Names unknown to the current model are silently ignored, so defaults cover multiple architectures. NemotronH note: ``experts`` matches NonGatedGroupedExperts inside LatentMoE; ``fc1_latent_proj``/``fc2_latent_proj`` adapt the latent up/down projections. Add ``in_proj``/``out_proj`` to also LoRA Mamba. |
| `model.lora.modules_to_save` | `list[str]` | `[]` | Module names or regex patterns to keep fully trainable (not freeze). Same matching rules as ``target_modules``. |

<a id="trainer-model-debug"></a>
#### `model.debug`

Debugging knobs for the model and distributed training.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.debug.num_layers` | `int | None` | `None` | Override the number of transformer layers (truncates the model). |
| `model.debug.random_init` | `bool` | `False` | Randomly initialize the model instead of loading weights. |
| `model.debug.force_balanced_routing` | `bool` | `False` | Replace MoE token-choice routing with a round-robin assignment so every expert sees an equal share. Intended for fake-data smoke tests where untrained routing would otherwise OOM under severe imbalance. Gating scores are still gathered from the override indices so the forward pass stays consistent. |

<a id="trainer-tokenizer"></a>
### `tokenizer`

| Field | Type | Default | Description |
|---|---|---|---|
| `tokenizer.name` | `str | None` | `None` | Tokenizer name or path. If None, the model's default tokenizer is used. |
| `tokenizer.trust_remote_code` | `bool | None` | `None` | Trust remote code when initializing the tokenizer. If None, inherits the model's ``trust_remote_code`` setting. |
| `tokenizer.chat_template` | `str | None` | `None` | Chat template for the tokenizer. Either a Jinja2 template string or a path to a template file. If None, the tokenizer's default chat template is used. |

<a id="trainer-data"></a>
### `data`

<a id="trainer-data-fake"></a>
#### `data.fake`

Use a fake data loader sampling random micro-batches (for debugging).

| Field | Type | Default | Description |
|---|---|---|---|
| `data.fake.batch_size` | `int` | `2` | _≥1._ Batch size of the fake data loader. |
| `data.fake.generate_samples` | `bool` | `False` | Generate separate samples and pack them into a single micro-batch instead of using random tensors. |

<a id="trainer-ckpt"></a>
### `ckpt`

Full training-state checkpoint configuration (model + optimizer + scheduler). If None, no resume-capable checkpoints are written.

| Field | Type | Default | Description |
|---|---|---|---|
| `ckpt.output_dir` | `Path | None` | `None` | Override directory for checkpoints and weights. If set, checkpoints and weight snapshots are written here instead of under the trainer ``output_dir`` — useful for writing large checkpoints to a separate storage volume. |
| `ckpt.interval` | `int | None` | `None` | _≥1._ Interval at which to save the training checkpoint. If None, only checkpoints at the end of training. |
| `ckpt.skip_gather_master_weights` | `bool` | `False` | Skip gathering and saving HF-compatible weight checkpoints. Useful for large models where the gather is expensive and only DCP checkpoints are needed. |
| `ckpt.weights_only` | `bool` | `False` | Save only weight checkpoints (no optimizer/scheduler state). Much faster and smaller than full checkpoints, but cannot resume training. |
| `ckpt.resume_step` | `int | None` | `None` | _≥-1._ Step to resume training from. None starts from scratch; ``-1`` restarts from the latest checkpoint available. |
| `ckpt.keep_last` | `int | None` | `None` | _≥1._ Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency. |
| `ckpt.keep_interval` | `int | None` | `None` | _≥1._ Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping. |
| `ckpt.skip_progress` | `bool` | `False` | Skip loading the progress from checkpoint. |
| `ckpt.skip_scheduler` | `bool` | `False` | Skip loading the scheduler from checkpoint. |
| `ckpt.skip_dataloader` | `bool` | `False` | Skip loading the dataloader from checkpoint. |
| `ckpt.skip_optimizer` | `bool` | `False` | Skip loading the optimizer state from checkpoint. |

<a id="trainer-ckpt-weights"></a>
#### `ckpt.weights`

Weight-checkpoint sub-configuration. If None, no HF-compatible weight checkpoints are written.

| Field | Type | Default | Description |
|---|---|---|---|
| `ckpt.weights.save_sharded` | `bool` | `True` | Save the weight checkpoint in sharded format. |
| `ckpt.weights.save_format` | `'safetensors' | 'torch'` | `'safetensors'` | Weight checkpoint serialization format. |
| `ckpt.weights.save_adapter_separately` | `bool` | `False` | Save LoRA adapters separately before merging into full model weights. |

<a id="trainer-log"></a>
### `log`

| Field | Type | Default | Description |
|---|---|---|---|
| `log.level` | `str` | `'info'` | Log level for the process. Defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``. |
| `log.vf_level` | `str` | `'info'` | Log level for the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) package. Defaults to ``$PRIME_VF_LOG_LEVEL`` if set, else ``info``. |
| `log.json_logging` | `bool` | `False` | Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.). |
| `log.log_data` | `bool` | `False` | Log the first data sample at startup. |
| `log.ranks_filter` | `list[int]` | `[0]` | Trainer ranks to show in console output. Passed to ``torchrun --local-ranks-filter``. |

<a id="trainer-wandb"></a>
### `wandb`

| Field | Type | Default | Description |
|---|---|---|---|
| `wandb.project` | `str` | `'prime-rl'` | W&B project to log to. |
| `wandb.entity` | `str | None` | `None` | W&B entity to log to. |
| `wandb.name` | `str | None` | `None` | W&B run name. |
| `wandb.group` | `str | None` | `None` | W&B group. |
| `wandb.tags` | `list[str] | None` | `None` | W&B tags attached to the run. |
| `wandb.offline` | `bool` | `False` | Run W&B in offline mode. |

<a id="trainer-bench"></a>
### `bench`

Benchmark-mode configuration. When set, ``max_steps`` is forced to 4 and fake data is used.

| Field | Type | Default | Description |
|---|---|---|---|
| `bench.output_json` | `Path | None` | `None` | Path to write benchmark results as JSON. If unset, results are only printed to the console. |

<a id="trainer-gc"></a>
### `gc`

Garbage collection config. Disables automatic GC and runs deterministic collections every N steps to avoid stragglers. Set to null to use Python's default GC behavior.

| Field | Type | Default | Description |
|---|---|---|---|
| `gc.interval` | `int` | `50` | _≥1._ Run garbage collection every N training steps. Disables Python's automatic GC so every rank collects together and one slow rank can't stall the others. |

<a id="trainer-heartbeat"></a>
### `heartbeat`

BetterStack heartbeat configuration for monitoring training progress.

| Field | Type | Default | Description |
|---|---|---|---|
| `heartbeat.url` | `str` | *required* | URL to send the heartbeat to. |

<a id="trainer-metrics-server"></a>
### `metrics_server`

Prometheus metrics server configuration. If set, exposes a ``/metrics`` endpoint for scraping.

| Field | Type | Default | Description |
|---|---|---|---|
| `metrics_server.port` | `int` | `8000` | _≥1, ≤65535._ Port to expose metrics and health endpoints on. |
| `metrics_server.host` | `str` | `'0.0.0.0'` | Host to bind the server to. |

<a id="trainer-experimental"></a>
### `experimental`

<a id="trainer-experimental-token-export"></a>
#### `experimental.token_export`

Opt-in per-token JSONL export for rollout debugging. When enabled, writes token ids and aligned trainer metrics after each forward pass.

<a id="trainer-loss"></a>
### `loss`

Loss config for rl-mode batches. opd and sft batches dispatch to their own loss fns unconditionally and do not read this.

Discriminated union — set `loss.type` to one of `default`, `custom` and provide the matching sub-fields.

<a id="trainer-loss-default"></a>
#### `loss.type = "default"` (DefaultLossConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `loss.type` | `'default'` | `'default'` |  |
| `loss.dppo_mask_low` | `float` | `0.2` | _≥0._ Lower DPPO masking threshold. |
| `loss.dppo_mask_high` | `float` | `0.2` | _≥0._ Upper DPPO masking threshold. |
| `loss.adv_tau` | `float` | `1.0` | _≥0._ Temperature for the advantage term. |
| `loss.kl_tau` | `float` | `0.001` | _≥0._ Temperature for the KL term. |

<a id="trainer-loss-custom"></a>
#### `loss.type = "custom"` (CustomLossConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `loss.type` | `'custom'` | `'custom'` |  |
| `loss.import_path` | `str` | *required* | Import path to the loss function (e.g. ``my_module.my_loss``). |
| `loss.kwargs` | `dict[str, Any]` | `{}` | Kwargs forwarded to the loss function. |

<a id="trainer-optim"></a>
### `optim`

Discriminated union — set `optim.type` to one of `sgd`, `adamw`, `muon`, `sign_sgd` and provide the matching sub-fields.

<a id="trainer-optim-sgd"></a>
#### `optim.type = "sgd"` (SGDConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `optim.type` | `'sgd'` | `'sgd'` |  |
| `optim.nesterov` | `bool` | `True` | Use Nesterov momentum. |
| `optim.momentum` | `float` | `0.9` | SGD momentum factor. |

<a id="trainer-optim-adamw"></a>
#### `optim.type = "adamw"` (AdamWConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `optim.type` | `'adamw'` | `'adamw'` |  |
| `optim.betas1` | `float` | `0.9` | _≥0._ Adam first-moment (β1) decay. |
| `optim.betas2` | `float` | `0.999` | _≥0._ Adam second-moment (β2) decay. |

<a id="trainer-optim-muon"></a>
#### `optim.type = "muon"` (MuonConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `optim.type` | `'muon'` | `'muon'` |  |
| `optim.mu` | `float` | `0.95` | _≥0._ Momentum factor for the Muon algorithm. |
| `optim.betas1` | `float` | `0.9` | _≥0._ β1 for the AdamW/Lion sub-optimizer used on non-Muon params. |
| `optim.betas2` | `float` | `0.95` | _≥0._ β2 for the AdamW/Lion sub-optimizer used on non-Muon params. |

<a id="trainer-optim-sign-sgd"></a>
#### `optim.type = "sign_sgd"` (SignSGDConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `1e-06` | _≥0._ Peak learning rate. |
| `optim.weight_decay` | `float` | `0.01` | _≥0._ L2 weight-decay coefficient. |
| `optim.max_norm` | `float | None` | `1.0` | _≥0._ Maximum gradient norm to clip to. If None, gradient clipping is disabled. |
| `optim.type` | `'sign_sgd'` | `'sign_sgd'` |  |

<a id="trainer-scheduler"></a>
### `scheduler`

Discriminated union — set `scheduler.type` to one of `constant`, `linear`, `cosine` and provide the matching sub-fields.

<a id="trainer-scheduler-constant"></a>
#### `scheduler.type = "constant"` (ConstantSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `scheduler.type` | `'constant'` | `'constant'` |  |

<a id="trainer-scheduler-linear"></a>
#### `scheduler.type = "linear"` (LinearSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `scheduler.type` | `'linear'` | `'linear'` |  |
| `scheduler.warmup_steps` | `int` | `10` | _≥0._ Warmup steps for the learning rate scheduler. |
| `scheduler.decay_steps` | `int` | `10` | _≥0._ Steps to decay the learning rate during the final portion of training. |
| `scheduler.min_lr` | `float` | `0.0` | _≥0._ Minimum learning rate to converge to. |

<a id="trainer-scheduler-cosine"></a>
#### `scheduler.type = "cosine"` (CosineSchedulerConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `scheduler.type` | `'cosine'` | `'cosine'` |  |
| `scheduler.warmup_steps` | `int` | `10` | _≥0._ Warmup steps for the learning rate scheduler. |
| `scheduler.min_lr` | `float` | `0.0` | _≥0._ Minimum learning rate to converge to. |

<a id="trainer-weight-broadcast"></a>
### `weight_broadcast`

Transport used to broadcast updated weights from trainer to inference.

Discriminated union — set `weight_broadcast.type` to one of `filesystem`, `nccl` and provide the matching sub-fields.

<a id="trainer-weight-broadcast-filesystem"></a>
#### `weight_broadcast.type = "filesystem"` (FileSystemWeightBroadcastConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `weight_broadcast.type` | `'filesystem'` | `'filesystem'` |  |
| `weight_broadcast.save_sharded` | `bool` | `True` | Save the weight checkpoint in sharded format. |
| `weight_broadcast.save_format` | `'safetensors' | 'torch'` | `'safetensors'` | Weight checkpoint serialization format. |

<a id="trainer-weight-broadcast-nccl"></a>
#### `weight_broadcast.type = "nccl"` (NCCLWeightBroadcastConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `weight_broadcast.type` | `'nccl'` | `'nccl'` |  |
| `weight_broadcast.host` | `str` | `'localhost'` | Host for the NCCL broadcast rendezvous. |
| `weight_broadcast.port` | `int` | `29501` | Port for the NCCL broadcast rendezvous. |
| `weight_broadcast.timeout` | `int` | `1200` | Timeout in seconds for the NCCL broadcast. |
| `weight_broadcast.inference_world_size` | `int` | `1` | Number of GPUs used for inference. |
| `weight_broadcast.quantize_in_weight_transfer` | `bool` | `False` | Use kernel-format FP8 quantized NCCL transfer for weight updates. When disabled, uses default HF checkpoint-format transfer. |

<a id="trainer-rollout-transport"></a>
### `rollout_transport`

Transport used to ship rollouts from orchestrator to trainer.

Discriminated union — set `rollout_transport.type` to one of `filesystem`, `zmq` and provide the matching sub-fields.

<a id="trainer-rollout-transport-filesystem"></a>
#### `rollout_transport.type = "filesystem"` (FileSystemTransportConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `rollout_transport.type` | `'filesystem'` | `'filesystem'` |  |

<a id="trainer-rollout-transport-zmq"></a>
#### `rollout_transport.type = "zmq"` (ZMQTransportConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `rollout_transport.type` | `'zmq'` | `'zmq'` |  |
| `rollout_transport.host` | `str` | `'localhost'` | Host address for ZMQ transport. |
| `rollout_transport.port` | `int` | `5555` | Base port for ZMQ transport. |
| `rollout_transport.hwm` | `int` | `10` | High-water mark (max in-flight messages per ZMQ socket). |

<a id="orchestrator"></a>
## `orchestrator` — Standalone orchestrator

The `orchestrator` entrypoint runs only the orchestrator process. It expects a separately-launched inference server to serve rollouts, and ships completed rollouts to a separately-launched trainer over the configured transport.

_Defined in_ `prime_rl.configs.orchestrator.OrchestratorConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `training_mode` | `'rl' | 'opd' | 'sft'` | `'rl'` | Training mode. ``rl``: student generates rollouts, no teacher. ``opd``: student generates rollouts, teacher computes logprobs (teacher_tau > 0). ``sft``: teacher generates rollouts, student inference pool used for evals and weight sync. |
| `advantage` | `DefaultAdvantageConfig | CustomAdvantageConfig | None` | `DefaultAdvantageConfig()` |  |
| `collect_inference_metrics` | `bool` | `True` | Collect inference-server metrics (requires wandb). |
| `output_dir` | `Path` | `'outputs/run_default'` | Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node. |
| `tasks_per_minute` | `int | None` | `None` | _≥1._ Rate limit per environment worker, in tasks per minute. Recommended for sandbox-backed environments to prevent sandbox-not-ready errors during autoscaling. With multiple workers, the effective total rate is ``workers × this value``. None disables rate limiting. |
| `batch_size` | `int | None` | `None` | _≥1._ Samples to train on per step (rollout-based batching). Set this OR ``token_batch_size``. |
| `token_batch_size` | `int | None` | `None` | _≥1._ Tokens to train on per step (token-based batching). Set this OR ``batch_size``. |
| `oversampling_factor` | `float | None` | `None` | _>0._ Rollout-mode batching only. Multiplier used to derive ``max_inflight_rollouts`` from ``batch_size`` when ``max_inflight_rollouts`` is unset. Values below 1.0 intentionally cap in-flight rollout capacity below ``batch_size``. |
| `max_inflight_rollouts` | `int | None` | `None` | _≥1._ Maximum number of rollouts kept in-flight. Required for token-based batching. With ``batch_size`` set, defaults to ``batch_size * oversampling_factor`` (or ``batch_size`` when ``oversampling_factor`` is unset). |
| `group_size` | `int` | `1` | _≥1._ Output sequences returned per example during training. |
| `seq_len` | `int` | `2048` | Training sequence length. Shorter samples are padded; longer samples are truncated. |
| `num_train_workers` | `int` | `1` | _≥1._ Training workers to use. |
| `max_steps` | `int | None` | `None` | Maximum training steps. If None, runs indefinitely. |
| `max_off_policy_steps` | `int` | `8` | _≥0._ Maximum policies allowed to generate a single rollout. Rollouts generated more than ``max_off_policy_steps`` ahead of training are discarded. Higher values yield better throughput at the cost of off-policy noise. |
| `max_async_level` | `int` | `1` | _≥0._ Maximum steps inference can be ahead of training. ``0`` degenerates to synchronous on-policy RL; ``≥1`` overlaps training and inference. |
| `strict_async_level` | `bool` | `False` | Strictly enforce ``max_async_level``. When True, the rollout policy is always exactly ``max_async_level`` steps ahead of training. When False, any policy within ``max_async_level`` steps is allowed (always uses the latest available policy). |
| `bench` | `bool` | `False` | Benchmark mode. Sets ``max_steps`` to 5, ``max_async_level`` to ~∞, and disables W&B. |
| `seed` | `int | None` | `42` | Random seed for the orchestrator. |
| `use_renderer` | `bool` | `True` | Use the renderer-backed TITO client (client-side tokenization via the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package, served by ``/v1/generate``). When True, the ``[orchestrator.renderer]`` block (name / tool_parser / reasoning_parser / pool_size) applies. Default for both text-only and VLM rollouts; VLMs require it. False falls back to MITO (``openai_chat_completions``). |
| `env_install_prerelease` | `bool` | `False` | Allow pre-release versions when installing environments (e.g. ``verifiers>=0.1.12.dev5``). Passes ``--prerelease`` to ``prime env install``. |

<a id="orchestrator-student"></a>
### `student`

Student rollout participant (model + client) — the model being trained.

<a id="orchestrator-student-model"></a>
#### `student.model`

| Field | Type | Default | Description |
|---|---|---|---|
| `student.model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `student.model.trust_remote_code` | `bool` | `False` | Trust remote code when initializing the tokenizer. |

<a id="orchestrator-student-model-vlm"></a>
##### `student.model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `student.model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `student.model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `student.model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="orchestrator-student-model-lora"></a>
##### `student.model.lora`

Per-run LoRA configuration. If None, LoRA is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `student.model.lora.name` | `str | None` | `None` | LoRA adapter name. If None, auto-generated from rank and alpha. |
| `student.model.lora.rank` | `int | None` | `None` | _≥1._ LoRA rank for this run. Must be ≤ trainer's max rank. If None, uses the trainer's rank. |
| `student.model.lora.alpha` | `float | None` | `None` | _≥0._ LoRA alpha for this run. If None, uses the trainer's alpha. |

<a id="orchestrator-student-client"></a>
#### `student.client`

| Field | Type | Default | Description |
|---|---|---|---|
| `student.client.timeout` | `int` | `1200` | Request timeout in seconds. |
| `student.client.connect_timeout` | `float` | `30.0` | TCP connect timeout in seconds for inference API requests. |
| `student.client.wait_for_ready_timeout` | `int` | `1800` | Seconds to wait at startup for the inference pool to become ready. Applies to both the static health check and elastic DNS-based discovery. |
| `student.client.base_url` | `list[str]` | `['http://localhost:8000/v1']` | Base URLs for the OpenAI API. With more than one URL, the client round-robins (chat) completion requests across all servers. Ignored when ``elastic`` is set. |
| `student.client.api_key_var` | `str` | `'VLLM_API_KEY'` | Environment variable name containing the API key, resolved via ``os.getenv``. Can be any string when the server is not protected by an API key; the same key is used for every URL. |
| `student.client.headers` | `dict[str, str]` | `{}` | Static headers sent with every request. |
| `student.client.headers_from_env` | `dict[str, str]` | `{}` | Maps HTTP header names to environment variable names; each entry is resolved via ``os.getenv`` and merged into request headers. e.g. ``{"X-Prime-Team-ID": "PRIME_TEAM_ID"}``. |
| `student.client.extra_headers_from_state` | `dict[str, str]` | `{}` | Maps HTTP header names to rollout-state field names. The header value is read from the rollout state dict on every request. e.g. ``{"X-Session-ID": "trajectory_id"}`` enables sticky routing at the inference router. |
| `student.client.skip_model_check` | `bool` | `False` | Skip checking that the model is available in the inference pool. Useful for external APIs or keys that do not expose ``/models``. |
| `student.client.dp_rank_count` | `int` | `1` | _≥1._ Number of data-parallel ranks behind each base URL. When > 1, each URL is expanded into ``dp_rank_count`` logical clients pinned via the ``X-data-parallel-rank`` header, so every request within a rollout hits the same DP engine and reuses KV cache. Auto-set from the inference config when using the RL entrypoint. |
| `student.client.admin_base_url` | `list[str] | None` | `None` | Separate base URLs for admin operations (weight updates, health checks). When set, admin clients bypass routers and hit each server directly — used in disaggregated P/D deployments where the router must not handle admin traffic. |
| `student.client.router_url` | `str | None` | `None` | vllm-router URL for load-aware inference routing. With elastic mode, inference requests go through the router while admin ops still hit discovered pods directly. |

<a id="orchestrator-student-client-elastic"></a>
##### `student.client.elastic`

Elastic inference pool config for DNS-based service discovery. When set, ``base_url`` is ignored and inference servers are discovered dynamically via DNS.

| Field | Type | Default | Description |
|---|---|---|---|
| `student.client.elastic.hostname` | `str` | *required* | DNS hostname that resolves to inference server IPs. |
| `student.client.elastic.port` | `int` | `8000` | Port that inference servers listen on. |
| `student.client.elastic.sync_interval` | `float` | `5.0` | Seconds between server discovery checks. |

<a id="orchestrator-teacher"></a>
### `teacher`

Teacher rollout participant (model + client). Role depends on ``training_mode``: ``opd`` — teacher computes logprobs; ``sft`` — teacher generates rollouts.

<a id="orchestrator-teacher-model"></a>
#### `teacher.model`

| Field | Type | Default | Description |
|---|---|---|---|
| `teacher.model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `teacher.model.trust_remote_code` | `bool` | `False` | Trust remote code when initializing the tokenizer. |

<a id="orchestrator-teacher-model-vlm"></a>
##### `teacher.model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `teacher.model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `teacher.model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `teacher.model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="orchestrator-teacher-model-lora"></a>
##### `teacher.model.lora`

Per-run LoRA configuration. If None, LoRA is disabled.

| Field | Type | Default | Description |
|---|---|---|---|
| `teacher.model.lora.name` | `str | None` | `None` | LoRA adapter name. If None, auto-generated from rank and alpha. |
| `teacher.model.lora.rank` | `int | None` | `None` | _≥1._ LoRA rank for this run. Must be ≤ trainer's max rank. If None, uses the trainer's rank. |
| `teacher.model.lora.alpha` | `float | None` | `None` | _≥0._ LoRA alpha for this run. If None, uses the trainer's alpha. |

<a id="orchestrator-teacher-client"></a>
#### `teacher.client`

| Field | Type | Default | Description |
|---|---|---|---|
| `teacher.client.timeout` | `int` | `1200` | Request timeout in seconds. |
| `teacher.client.connect_timeout` | `float` | `30.0` | TCP connect timeout in seconds for inference API requests. |
| `teacher.client.wait_for_ready_timeout` | `int` | `1800` | Seconds to wait at startup for the inference pool to become ready. Applies to both the static health check and elastic DNS-based discovery. |
| `teacher.client.base_url` | `list[str]` | `['http://localhost:8000/v1']` | Base URLs for the OpenAI API. With more than one URL, the client round-robins (chat) completion requests across all servers. Ignored when ``elastic`` is set. |
| `teacher.client.api_key_var` | `str` | `'VLLM_API_KEY'` | Environment variable name containing the API key, resolved via ``os.getenv``. Can be any string when the server is not protected by an API key; the same key is used for every URL. |
| `teacher.client.headers` | `dict[str, str]` | `{}` | Static headers sent with every request. |
| `teacher.client.headers_from_env` | `dict[str, str]` | `{}` | Maps HTTP header names to environment variable names; each entry is resolved via ``os.getenv`` and merged into request headers. e.g. ``{"X-Prime-Team-ID": "PRIME_TEAM_ID"}``. |
| `teacher.client.extra_headers_from_state` | `dict[str, str]` | `{}` | Maps HTTP header names to rollout-state field names. The header value is read from the rollout state dict on every request. e.g. ``{"X-Session-ID": "trajectory_id"}`` enables sticky routing at the inference router. |
| `teacher.client.skip_model_check` | `bool` | `False` | Skip checking that the model is available in the inference pool. Useful for external APIs or keys that do not expose ``/models``. |
| `teacher.client.dp_rank_count` | `int` | `1` | _≥1._ Number of data-parallel ranks behind each base URL. When > 1, each URL is expanded into ``dp_rank_count`` logical clients pinned via the ``X-data-parallel-rank`` header, so every request within a rollout hits the same DP engine and reuses KV cache. Auto-set from the inference config when using the RL entrypoint. |
| `teacher.client.admin_base_url` | `list[str] | None` | `None` | Separate base URLs for admin operations (weight updates, health checks). When set, admin clients bypass routers and hit each server directly — used in disaggregated P/D deployments where the router must not handle admin traffic. |
| `teacher.client.router_url` | `str | None` | `None` | vllm-router URL for load-aware inference routing. With elastic mode, inference requests go through the router while admin ops still hit discovered pods directly. |

<a id="orchestrator-teacher-client-elastic"></a>
##### `teacher.client.elastic`

Elastic inference pool config for DNS-based service discovery. When set, ``base_url`` is ignored and inference servers are discovered dynamically via DNS.

| Field | Type | Default | Description |
|---|---|---|---|
| `teacher.client.elastic.hostname` | `str` | *required* | DNS hostname that resolves to inference server IPs. |
| `teacher.client.elastic.port` | `int` | `8000` | Port that inference servers listen on. |
| `teacher.client.elastic.sync_interval` | `float` | `5.0` | Seconds between server discovery checks. |

<a id="orchestrator-train"></a>
### `train`

| Field | Type | Default | Description |
|---|---|---|---|
| `train.num_workers` | `int | 'auto'` | `'auto'` | Default worker processes for env servers. Can be overridden per env. |
| `train.max_retries` | `int` | `3` | _≥0._ Default retries for failed rollouts. Can be overridden per env. |

<a id="orchestrator-train-sampling"></a>
#### `train.sampling`

Shared training sampling configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `train.sampling.temperature` | `float` | `1.0` | _≥0._ Sampling temperature. |
| `train.sampling.repetition_penalty` | `float` | `1.0` | _≥0._ Repetition penalty. Values > 1.0 discourage repetition, < 1.0 encourage it, 1.0 disables. |
| `train.sampling.max_completion_tokens` | `int | None` | `None` | Maximum output tokens per turn. If None, generates until max context length or EOS. |
| `train.sampling.min_tokens` | `int` | `0` | _≥0._ Minimum output tokens per sequence. |
| `train.sampling.seed` | `int | None` | `None` | Random seed for sampling. If None, no seeding is used. |
| `train.sampling.extra_body` | `dict[str, Any]` | `{}` | Extra body forwarded with each request to the inference server. |

<a id="orchestrator-train-env"></a>
#### `train.env.<n>` (list item)

Training environments.

| Field | Type | Default | Description |
|---|---|---|---|
| `train.env.<n>.id` | `str` | `'reverse-text'` | Registered [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment ID (e.g. ``math-env``, ``primeintellect/math-env``). May include an ``@version`` suffix for installation. |
| `train.env.<n>.name` | `str | None` | `None` | Display name for this environment in logs, metrics, and buffer keys. Defaults to the ``id`` without ``@version``. Must be unique across all envs in the same group. |
| `train.env.<n>.args` | `dict` | `{}` | Keyword arguments forwarded to ``vf.load_environment``. See the environment's docstring for accepted args. |
| `train.env.<n>.extra_env_kwargs` | `dict[str, Any]` | `{}` | Extra kwargs passed to the env (e.g. ``seq_len``, ``max_total_completion_tokens``). Auto-populated by the orchestrator; user overrides are generally discouraged. The main use case is matching ``extra_env_kwargs`` when running an env in an isolated environment server. |
| `train.env.<n>.address` | `str | None` | `None` | ZMQ address of an external env server (e.g. ``tcp://host:5000``). When set, the orchestrator connects to this server instead of spawning one; when None, a subprocess env server is spawned automatically. |
| `train.env.<n>.num_workers` | `int | 'auto'` | `'auto'` | Worker processes for the spawned env server. ``auto`` scales to 1 worker per 256 concurrent rollouts. Ignored when ``address`` is set. |
| `train.env.<n>.ratio` | `float | None` | `None` | _>0._ Sampling weight for this environment in the buffer. When None for all envs, samples uniformly across all available problems. When set, must be set on all envs — values are relative weights normalized to probabilities (e.g. [1, 1] and [0.5, 0.5] are equivalent). |
| `train.env.<n>.max_retries` | `int` | `3` | _≥0._ Times the env server retries a failed rollout before returning an error. |
| `train.env.<n>.max_total_completion_tokens` | `int` | `-1` | Maximum total completion tokens across all turns in a multi-turn rollout. ``-1`` disables. Auto-populated into ``extra_env_kwargs``. |
| `train.env.<n>.timeout` | `float | None` | `None` | Per-rollout wall-clock timeout in seconds. None disables. |
| `train.env.<n>.state_columns` | `list[str]` | `[]` | Extra ``State`` fields to persist into the saved rollout records (in addition to the always-saved ``trajectory`` and ``sampling_args``). Values must be JSON-serializable. |

<a id="orchestrator-train-env-sampling"></a>
##### `train.env.<n>.sampling`

Per-env sampling overrides. Unset fields inherit from the group-level train sampling config.

| Field | Type | Default | Description |
|---|---|---|---|
| `train.env.<n>.sampling.temperature` | `float` | `1.0` | _≥0._ Sampling temperature. |
| `train.env.<n>.sampling.repetition_penalty` | `float` | `1.0` | _≥0._ Repetition penalty. Values > 1.0 discourage repetition, < 1.0 encourage it, 1.0 disables. |
| `train.env.<n>.sampling.max_completion_tokens` | `int | None` | `None` | Maximum output tokens per turn. If None, generates until max context length or EOS. |
| `train.env.<n>.sampling.min_tokens` | `int` | `0` | _≥0._ Minimum output tokens per sequence. |
| `train.env.<n>.sampling.seed` | `int | None` | `None` | Random seed for sampling. If None, no seeding is used. |
| `train.env.<n>.sampling.extra_body` | `dict[str, Any]` | `{}` | Extra body forwarded with each request to the inference server. |

<a id="orchestrator-tokenizer"></a>
### `tokenizer`

| Field | Type | Default | Description |
|---|---|---|---|
| `tokenizer.name` | `str | None` | `None` | Tokenizer name or path. If None, the model's default tokenizer is used. |
| `tokenizer.trust_remote_code` | `bool | None` | `None` | Trust remote code when initializing the tokenizer. If None, inherits the model's ``trust_remote_code`` setting. |
| `tokenizer.chat_template` | `str | None` | `None` | Chat template for the tokenizer. Either a Jinja2 template string or a path to a template file. If None, the tokenizer's default chat template is used. |

<a id="orchestrator-renderer"></a>
### `renderer`

Client-side renderer configuration. Only consumed when ``use_renderer=true``.

| Field | Type | Default | Description |
|---|---|---|---|
| `renderer.name` | `str` | `'auto'` | Renderer used for chat-template tokenization. One of: ``auto`` (detect from tokenizer), ``qwen3``, ``qwen3_vl``, ``qwen3.5``, ``glm5``, ``glm4.5``, ``minimax-m2``, ``deepseek_v3``, ``kimi_k2``, ``kimi_k25``, ``nemotron3``, ``gpt_oss``, ``default``. |
| `renderer.tool_parser` | `str | None` | `None` | Tool parser from [`renderers.parsers`](https://github.com/PrimeIntellect-ai/renderers). Only consumed by DefaultRenderer; model-specific renderers bake their own parsing in. Options: ``qwen3``, ``qwen3.5``, ``glm``, ``deepseek_v3``. |
| `renderer.reasoning_parser` | `str | None` | `None` | Reasoning parser from [`renderers.parsers`](https://github.com/PrimeIntellect-ai/renderers). Only consumed by DefaultRenderer. Options: ``think``. |
| `renderer.pool_size` | `int | None` | `None` | _≥1._ Number of renderer slots shared across concurrent rollouts. Bump for long multi-turn prompts where client-side jinja tokenization serializes. |
| `renderer.preserve_all_thinking` | `bool` | `False` | Re-emit every past-assistant turn's ``reasoning_content`` between ``<think>``/``</think>`` (or the model's equivalent), even when the chat template would drop it. Strict superset of preserve_thinking_between_tool_calls. |
| `renderer.preserve_thinking_between_tool_calls` | `bool` | `False` | Preserve past-assistant ``reasoning_content`` only inside the current tool cycle — the contiguous assistant→tool→…→assistant block after the most recent user message, when that block contains at least one tool response. A new user turn closes the block. |

<a id="orchestrator-optim"></a>
### `optim`

Per-run optimizer configuration for multi-run training.

| Field | Type | Default | Description |
|---|---|---|---|
| `optim.lr` | `float` | `0.0001` | _≥0._ Learning rate for this run (per-run override for multi-run training). |

<a id="orchestrator-eval"></a>
### `eval`

Evaluation configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `eval.num_examples` | `int` | `-1` | Default eval examples per environment. ``-1`` uses all. Can be overridden per env. |
| `eval.group_size` | `int` | `1` | _≥1._ Default rollouts per example. Can be overridden per env. |
| `eval.num_workers` | `int | 'auto'` | `'auto'` | Default worker processes for env servers. Can be overridden per env. |
| `eval.max_retries` | `int` | `3` | _≥0._ Default retries for failed rollouts. Can be overridden per env. |
| `eval.interval` | `int` | `100` | _≥1._ Step interval at which to evaluate the model. |
| `eval.eval_base_model` | `bool` | `True` | Evaluate the base model we are training on. |
| `eval.skip_eval_on_resume` | `bool` | `True` | When resuming the orchestrator from a checkpoint, skip the (potentially redundant) online eval that would otherwise run immediately at the resumed step. |
| `eval.cancel_inflight_rollouts_on_eval` | `bool` | `False` | Cancel in-flight training rollouts before starting online evals. Avoids congestion (no training + eval rollouts at the same time) at the cost of slower training steps as the pipeline has to refill after each eval. |

<a id="orchestrator-eval-sampling"></a>
#### `eval.sampling`

Shared eval sampling configuration; can differ from training sampling.

| Field | Type | Default | Description |
|---|---|---|---|
| `eval.sampling.temperature` | `float | None` | `None` | _≥0._ Sampling temperature. None defers to the inference server default. |
| `eval.sampling.repetition_penalty` | `float | None` | `None` | _≥0._ Repetition penalty. None defers to the inference server default. |
| `eval.sampling.top_p` | `float | None` | `None` | Nucleus sampling threshold. None defers to the inference server default. |
| `eval.sampling.top_k` | `int | None` | `None` | Top-k sampling. None defers to the inference server default. |
| `eval.sampling.min_p` | `float | None` | `None` | _≥0._ Min-p sampling threshold. None defers to the inference server default. |
| `eval.sampling.max_completion_tokens` | `int | None` | `None` | Maximum output tokens per turn. None defers to the inference server default. |
| `eval.sampling.min_tokens` | `int | None` | `None` | _≥0._ Minimum output tokens per sequence. None defers to the inference server default. |
| `eval.sampling.reasoning_effort` | `'minimal' | 'low' | 'medium' | 'high' | None` | `None` | Reasoning effort constraint for reasoning models. |
| `eval.sampling.seed` | `int | None` | `None` | Random seed for sampling. None means no seeding. |
| `eval.sampling.extra_body` | `dict[str, Any]` | `{}` | Extra body parameters forwarded to the inference server. |

<a id="orchestrator-eval-env"></a>
#### `eval.env.<n>` (list item)

Evaluation environments.

| Field | Type | Default | Description |
|---|---|---|---|
| `eval.env.<n>.id` | `str` | `'reverse-text'` | Registered [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment ID (e.g. ``math-env``, ``primeintellect/math-env``). May include an ``@version`` suffix for installation. |
| `eval.env.<n>.name` | `str | None` | `None` | Display name for this environment in logs, metrics, and buffer keys. Defaults to the ``id`` without ``@version``. Must be unique across all envs in the same group. |
| `eval.env.<n>.args` | `dict` | `{}` | Keyword arguments forwarded to ``vf.load_environment``. See the environment's docstring for accepted args. |
| `eval.env.<n>.extra_env_kwargs` | `dict[str, Any]` | `{}` | Extra kwargs passed to the env (e.g. ``seq_len``, ``max_total_completion_tokens``). Auto-populated by the orchestrator; user overrides are generally discouraged. The main use case is matching ``extra_env_kwargs`` when running an env in an isolated environment server. |
| `eval.env.<n>.address` | `str | None` | `None` | ZMQ address of an external env server (e.g. ``tcp://host:5000``). When set, the orchestrator connects to this server instead of spawning one; when None, a subprocess env server is spawned automatically. |
| `eval.env.<n>.num_workers` | `int | 'auto'` | `'auto'` | Worker processes for the spawned env server. ``auto`` scales to 1 worker per 256 concurrent rollouts. Ignored when ``address`` is set. |
| `eval.env.<n>.ratio` | `float | None` | `None` | _>0._ Sampling weight for this environment in the buffer. When None for all envs, samples uniformly across all available problems. When set, must be set on all envs — values are relative weights normalized to probabilities (e.g. [1, 1] and [0.5, 0.5] are equivalent). |
| `eval.env.<n>.max_retries` | `int` | `3` | _≥0._ Times the env server retries a failed rollout before returning an error. |
| `eval.env.<n>.max_total_completion_tokens` | `int` | `-1` | Maximum total completion tokens across all turns in a multi-turn rollout. ``-1`` disables. Auto-populated into ``extra_env_kwargs``. |
| `eval.env.<n>.timeout` | `float | None` | `None` | Per-rollout wall-clock timeout in seconds. None disables. |
| `eval.env.<n>.state_columns` | `list[str]` | `[]` | Extra ``State`` fields to persist into the saved rollout records (in addition to the always-saved ``trajectory`` and ``sampling_args``). Values must be JSON-serializable. |
| `eval.env.<n>.num_examples` | `int` | `-1` | Eval examples to sample from the dataset. ``-1`` uses all available examples. |
| `eval.env.<n>.group_size` | `int` | `1` | _≥1._ Rollouts generated per example. Used for pass@k estimation (e.g. ``group_size=8`` enables pass@1 through pass@8). |
| `eval.env.<n>.interval` | `int` | `100` | _≥1._ Per-env eval interval. If unset, inherits from the group-level eval interval. |

<a id="orchestrator-eval-env-sampling"></a>
##### `eval.env.<n>.sampling`

Per-env sampling overrides. Unset fields inherit from the group-level eval sampling config.

| Field | Type | Default | Description |
|---|---|---|---|
| `eval.env.<n>.sampling.temperature` | `float | None` | `None` | _≥0._ Sampling temperature. None defers to the inference server default. |
| `eval.env.<n>.sampling.repetition_penalty` | `float | None` | `None` | _≥0._ Repetition penalty. None defers to the inference server default. |
| `eval.env.<n>.sampling.top_p` | `float | None` | `None` | Nucleus sampling threshold. None defers to the inference server default. |
| `eval.env.<n>.sampling.top_k` | `int | None` | `None` | Top-k sampling. None defers to the inference server default. |
| `eval.env.<n>.sampling.min_p` | `float | None` | `None` | _≥0._ Min-p sampling threshold. None defers to the inference server default. |
| `eval.env.<n>.sampling.max_completion_tokens` | `int | None` | `None` | Maximum output tokens per turn. None defers to the inference server default. |
| `eval.env.<n>.sampling.min_tokens` | `int | None` | `None` | _≥0._ Minimum output tokens per sequence. None defers to the inference server default. |
| `eval.env.<n>.sampling.reasoning_effort` | `'minimal' | 'low' | 'medium' | 'high' | None` | `None` | Reasoning effort constraint for reasoning models. |
| `eval.env.<n>.sampling.seed` | `int | None` | `None` | Random seed for sampling. None means no seeding. |
| `eval.env.<n>.sampling.extra_body` | `dict[str, Any]` | `{}` | Extra body parameters forwarded to the inference server. |

<a id="orchestrator-buffer"></a>
### `buffer`

| Field | Type | Default | Description |
|---|---|---|---|
| `buffer.seed` | `int | None` | `None` | Random seed for the buffer. When set, sampling from the buffer is deterministic. |
| `buffer.easy_threshold` | `float | None` | `None` | Average-reward threshold above which a problem is classified ``easy``. |
| `buffer.hard_threshold` | `float | None` | `None` | Average-reward threshold below which a problem is classified ``hard``. |
| `buffer.easy_fraction` | `float` | `0.0` | _≥0, ≤1._ Fraction of easy problems to convert to ``normal`` when resuming or starting training. Only problems with difficulty ``normal`` are sampled. |
| `buffer.hard_fraction` | `float` | `0.0` | _≥0, ≤1._ Fraction of hard problems to convert to ``normal`` when resuming or starting training. Only problems with difficulty ``normal`` are sampled. |
| `buffer.online_difficulty_filtering` | `bool` | `False` | Filter rollouts based on difficulty. When True, rollouts with average reward 0.0 or 1.0 are not added to the buffer. |
| `buffer.hash_keys` | `list[str]` | `['env_name', 'prompt']` | _len ≥ 1._ Keys used to compute example hashes. Used to match examples from buffer checkpoints and determine buffer resume behavior. |

<a id="orchestrator-log"></a>
### `log`

| Field | Type | Default | Description |
|---|---|---|---|
| `log.level` | `str` | `'info'` | Log level for the process. Defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``. |
| `log.vf_level` | `str` | `'info'` | Log level for the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) package. Defaults to ``$PRIME_VF_LOG_LEVEL`` if set, else ``info``. |
| `log.json_logging` | `bool` | `False` | Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.). |
| `log.log_data` | `bool` | `False` | Log the first data sample at startup. |

<a id="orchestrator-wandb"></a>
### `wandb`

| Field | Type | Default | Description |
|---|---|---|---|
| `wandb.project` | `str` | `'prime-rl'` | W&B project to log to. |
| `wandb.entity` | `str | None` | `None` | W&B entity to log to. |
| `wandb.name` | `str | None` | `None` | W&B run name. |
| `wandb.group` | `str | None` | `None` | W&B group. |
| `wandb.tags` | `list[str] | None` | `None` | W&B tags attached to the run. |
| `wandb.offline` | `bool` | `False` | Run W&B in offline mode. |

<a id="orchestrator-wandb-log-extras"></a>
#### `wandb.log_extras`

Extras logging configuration. If None, no extras are logged.

| Field | Type | Default | Description |
|---|---|---|---|
| `wandb.log_extras.samples` | `bool` | `True` | Log prompt/response samples. |
| `wandb.log_extras.distributions` | `bool` | `True` | Log distributions (rewards, advantages, etc.). |
| `wandb.log_extras.interval` | `int` | `10` | _≥1._ Step interval between extras logs. |
| `wandb.log_extras.sample_ratio` | `float | None` | `None` | _≥0.0, ≤1.0._ Fraction of rollouts to log per step. The effective cap is ``len(rollouts) * sample_ratio``; 1.0 = all, 0.5 = half, 0.0 = none. |

<a id="orchestrator-prime-monitor"></a>
### `prime_monitor`

| Field | Type | Default | Description |
|---|---|---|---|
| `prime_monitor.base_url` | `str` | `'https://api.primeintellect.ai/api/v1/rft'` | Base URL for the Prime Intellect monitoring API. |
| `prime_monitor.api_key_var` | `str` | `'PRIME_API_KEY'` | Environment variable name containing the Prime Intellect API key, resolved via ``os.getenv``. |
| `prime_monitor.run_name` | `str | None` | `None` | Run name shown on the platform. Defaults to the W&B run name when set, otherwise the platform auto-generates one. |
| `prime_monitor.team_id` | `str | None` | `None` | Team ID to associate the run with. |
| `prime_monitor.frontend_url` | `str | None` | `None` | Frontend base URL used for the dashboard link printed after registration. Defaults to the Prime CLI frontend URL when unset. |

<a id="orchestrator-prime-monitor-log-extras"></a>
#### `prime_monitor.log_extras`

Extras logging configuration. If None, no extras are logged.

| Field | Type | Default | Description |
|---|---|---|---|
| `prime_monitor.log_extras.samples` | `bool` | `True` | Log prompt/response samples. |
| `prime_monitor.log_extras.distributions` | `bool` | `True` | Log distributions (rewards, advantages, etc.). |
| `prime_monitor.log_extras.interval` | `int` | `10` | _≥1._ Step interval between extras logs. |
| `prime_monitor.log_extras.sample_ratio` | `float | None` | `None` | _≥0.0, ≤1.0._ Fraction of rollouts to log per step. The effective cap is ``len(rollouts) * sample_ratio``; 1.0 = all, 0.5 = half, 0.0 = none. |

<a id="orchestrator-ckpt"></a>
### `ckpt`

Checkpoint configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `ckpt.interval` | `int | None` | `None` | _≥1._ Step interval at which to save the orchestrator checkpoint. |
| `ckpt.resume_step` | `int | None` | `None` | _≥-1._ Step to resume the orchestrator from. None starts from scratch; ``-1`` resumes from the latest checkpoint available. |
| `ckpt.wait_for_weights_timeout` | `int | None` | `None` | _≥1._ When resuming, wait up to this many seconds for the weight directory to appear. Useful when the orchestrator restarts while the trainer is still saving weights. If None, fail immediately when weights are not found. |
| `ckpt.keep_last` | `int | None` | `None` | _≥1._ Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency. |
| `ckpt.keep_interval` | `int | None` | `None` | _≥1._ Keep checkpoints at every N steps permanently (e.g. ``keep_interval=100`` keeps step 100, 200, ...). If None, no interval-based keeping. |
| `ckpt.skip_progress` | `bool` | `False` | Skip loading the progress from checkpoint. |
| `ckpt.skip_buffer` | `bool` | `False` | Skip loading the buffer from checkpoint. |

<a id="orchestrator-heartbeat"></a>
### `heartbeat`

BetterStack heartbeat configuration for monitoring training progress.

| Field | Type | Default | Description |
|---|---|---|---|
| `heartbeat.url` | `str` | *required* | URL to send the heartbeat to. |

<a id="orchestrator-experimental"></a>
### `experimental`

<a id="orchestrator-filters"></a>
### `filters.<n>` (list item)

Rollout filters. Each filter can ``monitor`` (default) or ``enforce`` (skip rollouts).

Discriminated list-item union — set `filters.<n>.type` to one of `gibberish`, `repetition`, `zero_advantage` and provide the matching sub-fields.

<a id="orchestrator-filters-gibberish"></a>
#### `filters.<n>.type = "gibberish"` (GibberishFilterConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `filters.<n>.type` | `'gibberish'` | `'gibberish'` |  |
| `filters.<n>.enforce` | `bool` | `False` | When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics. |
| `filters.<n>.token_id_threshold` | `int` | `100000` | Token IDs above this are candidates for gibberish. BPE tokens are sorted by merge order. |
| `filters.<n>.logprob_offset` | `float` | `2.0` | Offset from uniform-distribution logprob. Threshold = ``-log(vocab_size) - logprob_offset``. |

<a id="orchestrator-filters-repetition"></a>
#### `filters.<n>.type = "repetition"` (RepetitionFilterConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `filters.<n>.type` | `'repetition'` | `'repetition'` |  |
| `filters.<n>.enforce` | `bool` | `False` | When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics. |
| `filters.<n>.window` | `int` | `3000` | _≥1._ Consecutive high-probability steps required to flag the rollout. |
| `filters.<n>.prob_threshold` | `float` | `0.99` | _>0, ≤1._ Tokens sampled with probability above this are considered repetitive. Consecutive such tokens count toward the window. |

<a id="orchestrator-filters-zero-advantage"></a>
#### `filters.<n>.type = "zero_advantage"` (ZeroAdvantageFilterConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `filters.<n>.type` | `'zero_advantage'` | `'zero_advantage'` |  |
| `filters.<n>.enforce` | `bool` | `True` | When True, skip detected rollouts entirely so they are not sent to the trainer. When False, only track detection metrics. |

<a id="orchestrator-weight-broadcast"></a>
### `weight_broadcast`

Transport used to receive updated weights from the trainer.

Discriminated union — set `weight_broadcast.type` to one of `filesystem`, `nccl` and provide the matching sub-fields.

<a id="orchestrator-weight-broadcast-filesystem"></a>
#### `weight_broadcast.type = "filesystem"` (FileSystemWeightBroadcastConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `weight_broadcast.type` | `'filesystem'` | `'filesystem'` |  |

<a id="orchestrator-weight-broadcast-nccl"></a>
#### `weight_broadcast.type = "nccl"` (NCCLWeightBroadcastConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `weight_broadcast.type` | `'nccl'` | `'nccl'` |  |
| `weight_broadcast.host` | `str` | `'localhost'` | Host for the NCCL broadcast rendezvous. |
| `weight_broadcast.port` | `int` | `29501` | Port for the NCCL broadcast rendezvous. |
| `weight_broadcast.timeout` | `int` | `1200` | Timeout in seconds for the NCCL broadcast. |
| `weight_broadcast.quantize_in_weight_transfer` | `bool` | `False` | Use kernel-format FP8 quantized NCCL transfer for weight updates. |
| `weight_broadcast.inference_world_size` | `int` | `1` | _≥1._ Total inference GPUs across all servers. Used by ``init_nccl_broadcast`` to compute per-server rank offsets. |

<a id="orchestrator-rollout-transport"></a>
### `rollout_transport`

Transport used to ship rollouts from orchestrator to trainer.

Discriminated union — set `rollout_transport.type` to one of `filesystem`, `zmq` and provide the matching sub-fields.

<a id="orchestrator-rollout-transport-filesystem"></a>
#### `rollout_transport.type = "filesystem"` (FileSystemTransportConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `rollout_transport.type` | `'filesystem'` | `'filesystem'` |  |

<a id="orchestrator-rollout-transport-zmq"></a>
#### `rollout_transport.type = "zmq"` (ZMQTransportConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `rollout_transport.type` | `'zmq'` | `'zmq'` |  |
| `rollout_transport.host` | `str` | `'localhost'` | Host address for ZMQ transport. |
| `rollout_transport.port` | `int` | `5555` | Base port for ZMQ transport. |
| `rollout_transport.hwm` | `int` | `10` | High-water mark (max in-flight messages per ZMQ socket). |

<a id="inference"></a>
## `inference` — Standalone vLLM server

The `inference` entrypoint launches a vLLM server (or a disaggregated prefill/decode pair) that serves OpenAI-compatible completions to the orchestrator.

_Defined in_ `prime_rl.configs.inference.InferenceConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `enable_lora` | `bool` | `False` | Enable LoRA. Forwarded as ``--enable-lora``. |
| `max_loras` | `int` | `8` | Maximum number of LoRAs. Forwarded as ``--max-loras``. |
| `max_cpu_loras` | `int` | `100` | Maximum number of LoRAs on CPU. Forwarded as ``--max-cpu-loras``. |
| `max_lora_rank` | `int | None` | `None` | Maximum LoRA rank. Forwarded as ``--max-lora-rank``. |
| `lora_target_modules` | `list[str] | None` | `None` | LoRA target modules. Forwarded as ``--lora-target-modules``. |
| `enable_prefix_caching` | `bool | None` | `None` | Enable prefix caching. Forwarded as ``--enable-prefix-caching``. |
| `gpu_memory_utilization` | `float` | `0.9` | GPU memory utilization. Forwarded as ``--gpu-memory-utilization``. |
| `api_server_count` | `int` | `1` | _≥0._ API servers to run. Forwarded as ``--api-server-count``. Set to 0 for headless mode. |
| `data_parallel_size_local` | `int | None` | `None` | _≥1._ Data parallel replicas to run on this node. Forwarded as ``--data-parallel-size-local``. |
| `data_parallel_rpc_port` | `int` | `13345` | _≥1, ≤65535._ RPC port for data parallel communication. Forwarded as ``--data-parallel-rpc-port``. |
| `seed` | `int` | `0` | Seed the inference components. Forwarded as ``--seed``. |
| `enable_expert_parallel` | `bool` | `False` | Enable expert parallelism for MoE models. Forwarded as ``--enable-expert-parallel``. |
| `all2all_backend` | `'allgather_reducescatter' | 'deepep_high_throughput' | 'deepep_low_latency' | 'flashinfer_nvlink_one_sided' | 'flashinfer_nvlink_two_sided'` | `'allgather_reducescatter'` | All-to-all backend for expert-parallel communication. Forwarded as ``--all2all-backend``. |
| `enable_eplb` | `bool` | `False` | Enable expert parallel load balancer (EPLB). Forwarded as ``--enable-eplb``. |
| `enable_dbo` | `bool` | `False` | Enable dual batch overlap (DBO). Forwarded as ``--enable-dbo``. |
| `use_deep_gemm` | `bool` | `False` | Force DeepGEMM FP8 kernels via ``VLLM_USE_DEEP_GEMM=1``. Only works with per-tensor FP8 quantization (e.g. GLM-5-FP8). |
| `enable_return_routed_experts` | `bool` | `False` | Return routed experts in responses. Forwarded as ``--enable-return-routed-experts``. |
| `enable_fp32_lm_head` | `bool` | `False` | Run the lm_head projection in fp32 via a native bf16×bf16 → fp32 GEMM (``torch.mm`` with ``out_dtype=torch.float32``). Stabilizes logprob precision under FP8/bf16 inference, matching SGLang's ``--enable-fp32-lm-head``. Implemented as a monkey-patch over vLLM's LogitsProcessor, activated by setting ``additional_config["fp32_lm_head"] = True`` on the vLLM config. |
| `vllm_extra` | `dict[str, Any]` | `{}` | Extra arguments forwarded to vLLM. Applied as attributes on the vLLM namespace after config translation. |
| `output_dir` | `Path` | `'outputs'` | Directory for SLURM logs and generated scripts. |
| `dry_run` | `bool` | `False` | Only validate and dump resolved configs, then exit early. |

<a id="inference-server"></a>
### `server`

| Field | Type | Default | Description |
|---|---|---|---|
| `server.host` | `str | None` | `None` | Host to bind to. |
| `server.port` | `int` | `8000` | Port to bind to. |
| `server.liveness_timeout_seconds` | `float` | `30.0` | _>0._ Timeout in seconds for the ``/liveness`` endpoint's internal vLLM worker RPC. With Kubernetes liveness probes, keep the probe ``timeoutSeconds`` at least this high. |

<a id="inference-model"></a>
### `model`

| Field | Type | Default | Description |
|---|---|---|---|
| `model.name` | `str` | `'Qwen/Qwen3-0.6B'` | HF model name or local path. |
| `model.trust_remote_code` | `bool` | `False` | Trust remote code. Forwarded to vLLM engine init. |
| `model.dtype` | `'auto' | 'float16' | 'bfloat16' | 'float32'` | `'auto'` | dtype for model weights and activations. ``auto`` uses FP16 for FP32/FP16 models and BF16 for BF16 models. Forwarded as ``--dtype``. |
| `model.max_model_len` | `int | None` | `None` | Maximum model context length. If None, uses the model config's value. Forwarded as ``--max-model-len``. |
| `model.enforce_eager` | `bool` | `False` | Enforce eager mode. When False, PyTorch eager and cuda graphs run hybrid for maximum performance. Forwarded as ``--enforce-eager``. |
| `model.chat_template` | `str | None` | `None` | Chat template — a Jinja2 template string or path to a template file. Forwarded as ``--chat-template``. If None, uses the model's default. |
| `model.tool_call_parser` | `str | None` | `'auto'` | Tool-call parser. Forwarded as ``--tool-call-parser``. Set to ``"auto"`` (default) to detect from the model name, or ``None`` to disable. |
| `model.reasoning_parser` | `str | None` | `'auto'` | Parser for extracting reasoning content from model outputs. Forwarded as ``--reasoning-parser``. Set to ``"auto"`` (default) to detect from the model name, or ``None`` to disable. |
| `model.rope_scaling` | `dict[str, Any] | str | None` | `None` | RoPE scaling configuration as a dict (e.g. ``{rope_type="yarn", factor=4.0, original_max_position_embeddings=32768}``). Forwarded as ``--rope-scaling``. |

<a id="inference-model-vlm"></a>
#### `model.vlm`

VLM configuration. Setting this enables vision-language model support.

| Field | Type | Default | Description |
|---|---|---|---|
| `model.vlm.vision_encoder_attr` | `str` | *required* | Dotted attribute path to the vision encoder module (e.g. ``model.visual``). |
| `model.vlm.language_model_attr` | `str` | *required* | Dotted attribute path to the language model module (e.g. ``model.language_model``). |
| `model.vlm.freeze_vision_encoder` | `bool` | `True` | Freeze the vision encoder. When False, it is trainable and FSDP-sharded per-block. No effect with LoRA (LoRA freezes all non-adapter parameters). |

<a id="inference-parallel"></a>
### `parallel`

Multi-node and multi-GPU parallelism (TP, DP, PP).

| Field | Type | Default | Description |
|---|---|---|---|
| `parallel.tp` | `int` | `1` | Tensor parallel size. Forwarded to vLLM as ``--tensor-parallel-size``. |
| `parallel.dp` | `int` | `1` | _≥1._ Data parallel size. Forwarded to vLLM as ``--data-parallel-size``. |

<a id="inference-weight-broadcast"></a>
### `weight_broadcast`

| Field | Type | Default | Description |
|---|---|---|---|
| `weight_broadcast.type` | `'nccl' | 'filesystem'` | `'filesystem'` | Weight broadcast transport. |

<a id="inference-kv-cache-offload"></a>
### `kv_cache_offload`

CPU KV cache offload for inference workers. Standard inference uses vLLM's ``OffloadingConnector``. Disaggregated P/D deployments combine it with NIXL through ``MultiConnector`` in the SLURM launcher.

| Field | Type | Default | Description |
|---|---|---|---|
| `kv_cache_offload.cpu_bytes` | `int` | `1000000000` | _>0._ CPU bytes available for KV cache offloading per worker. |

<a id="inference-slurm"></a>
### `slurm`

SLURM configuration. When set, the run is submitted as a SLURM job instead of running locally.

| Field | Type | Default | Description |
|---|---|---|---|
| `slurm.job_name` | `str` | `'prime-rl'` | SLURM job name. |
| `slurm.project_dir` | `Path` | `'.'` | Path to the project root, used to source .env, activate .venv, and run uv sync. |
| `slurm.template_path` | `Path | None` | `None` | SLURM template file. If None, uses the bundled single-node or multi-node template. |
| `slurm.partition` | `str` | `'cluster'` | SLURM partition (#SBATCH --partition). |
| `slurm.nodelist` | `str | None` | `None` | Comma-separated list of specific nodes to run on (#SBATCH --nodelist). |
| `slurm.exclude` | `str | None` | `None` | Comma-separated list of nodes to exclude (#SBATCH --exclude). |
| `slurm.account` | `str | None` | `None` | SLURM account to charge (#SBATCH --account). |
| `slurm.time` | `str | None` | `None` | Maximum wall time, e.g. '24:00:00' or '7-00:00:00' (#SBATCH --time). |
| `slurm.pre_run_command` | `str | None` | `None` | Shell command to run on the head node after cd, .env sourcing, and venv activation. Useful for cleanup like ``sudo pkill -f vllm``; wrap with ``srun bash -c '...'`` to fan out to all nodes. |

<a id="inference-experimental"></a>
### `experimental`

<a id="inference-deployment"></a>
### `deployment`

Discriminated union — set `deployment.type` to one of `single_node`, `multi_node`, `disaggregated` and provide the matching sub-fields.

<a id="inference-deployment-single-node"></a>
#### `deployment.type = "single_node"` (SingleNodeInferenceDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `deployment.type` | `'single_node'` | `'single_node'` |  |

<a id="inference-deployment-multi-node"></a>
#### `deployment.type = "multi_node"` (MultiNodeInferenceDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `deployment.type` | `'multi_node'` | `'multi_node'` |  |
| `deployment.num_nodes` | `int` | `2` | _≥1._ Inference nodes. |
| `deployment.router_port` | `int` | `8000` | Port for the vllm-router. |
| `deployment.backend_port` | `int` | `8100` | Port for vLLM backend instances. |
| `deployment.router_policy` | `str` | `'consistent_hash'` | vllm-router routing policy (e.g. ``consistent_hash``, ``round_robin``). |

<a id="inference-deployment-disaggregated"></a>
#### `deployment.type = "disaggregated"` (DisaggregatedInferenceDeploymentConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `deployment.gpus_per_node` | `int` | `8` | GPUs per node. |
| `deployment.type` | `'disaggregated'` | `'disaggregated'` |  |
| `deployment.num_prefill_nodes` | `int` | `1` | _≥1._ Total prefill nodes. |
| `deployment.num_decode_nodes` | `int` | `1` | _≥1._ Total decode nodes. |
| `deployment.num_prefill_replicas` | `int` | `1` | _≥1._ Independent prefill vLLM instances. Must evenly divide ``num_prefill_nodes``. |
| `deployment.num_decode_replicas` | `int` | `1` | _≥1._ Independent decode vLLM instances. Must evenly divide ``num_decode_nodes``. |
| `deployment.router_port` | `int` | `8000` | Port for the vllm-router on each replica. |
| `deployment.prefill_port` | `int` | `8100` | Port for prefill vLLM instances. |
| `deployment.decode_port` | `int` | `8200` | Port for decode vLLM instances. |
| `deployment.router_policy` | `str` | `'consistent_hash'` | vllm-router routing policy (e.g. ``consistent_hash``, ``round_robin``). |
| `deployment.prefill_env_overrides` | `dict[str, str]` | `{}` | Extra environment variables exported only on prefill nodes. |
| `deployment.decode_env_overrides` | `dict[str, str]` | `{}` | Extra environment variables exported only on decode nodes. |

## About this page

Each entrypoint section walks its config tree top-down. Nested sub-configs
appear under headings named after their dotted path (e.g. `trainer.model.ac`).
List-typed sub-configs (e.g. `[[orchestrator.train.env]]`) appear under
headings with a `<n>` index placeholder — that's the CLI form too
(`--orchestrator.train.env.0.id ...`). Discriminated unions (loss, advantage,
scheduler, optimizer, …) document each variant in turn — set the `type` field
to pick one.

To regenerate, run from the project root:

```bash
uv run python scripts/generate_docs_reference.py
```

For conceptual context behind these knobs, see
[Configuration](configuration.md), [Training](training.md),
[Scaling](scaling.md), [Algorithms](algorithms.md), and [Advanced](advanced.md).
