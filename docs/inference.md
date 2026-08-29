# Inference

This page covers the inference configuration and the supported features/deployment shapes. It covers how to scale the inference server from a single GPU to 1000s of GPUs that run agentic workloads at the speed of light with all the bells and whistles configured.

## Table of Contents

- [Overview](#overview)
- [Single-Node](#single-node)
- [Multi-Node](#multi-node)
    - [Multi-replica](#multi-replica)
    - [Wide-EP](#wide-ep)
- [P/D Disaggregation](#pd-disaggregation)
- [Router](#router)
    - [Routing policies](#routing-policies)
- [Adaptive Concurrency](#adaptive-concurrency)
- [Advanced Configuration](#advanced-configuration)
    - [KV Cache Offload](#kv-cache-offload)
    - [Optimized P/D disaggregation deployment](#optimized-pd-disaggregation-deployment)
    - [Other vLLM features](#other-vllm-features)
    - [Router Replay](#router-replay)


## Overview

`prime-rl` chooses to use `vLLM` as the inference engine. We aim to stay up-to-date with the latest vLLM features, being at-most 1 version behind the latest stable release. This allows us to use the latest features from vLLM as soon as they are released - such as router replay, CPU KV cache offload, and more.

We support 3 distinct deployment shapes:
- [Single-Node](#single-node) - Runs the inference server on a single node. Useful for debugging, small scale experiments or smaller models. The default deployment shape.
- [Multi-Node](#multi-node) - Runs the inference server on multiple nodes. Useful for large scale experiments or larger models, where latency is not a concern - i.e. single turn inference, long context inference, etc.
- [Disaggregated](#pd-disaggregation) - Runs the inference server on multiple nodes, but disaggregates the prefill and decode stages. Useful for large scale experiments or larger models, where latency is a concern and multi-node deployment creates very high E2E rollout latency, such as agentic workflows.

Most of the features are supported for all deployment shapes, with few exceptions. These exceptions are rejected on validation.

Every deployment shape has the same client-facing layout: a single global router listens on `inference.server.port` and fronts all vLLM engines, which listen on `inference.backend_port` (+ rank offset). Clients always talk to one URL, regardless of how many engines run behind it.

You can select the deployment shape with `InferenceDeploymentConfig` in your config file. This is a config-field that allows you to set the deployment shape and topology knobs such as `num_nodes` and `num_replicas`.

```toml
[inference.deployment]
type = "single_node" # or "multi_node" or "disaggregated"
```

To configure the inference server, you can use the `InferenceConfig` field. This is a config-field that allows you to set the inference server-specific knobs. Most of these are supported for all deployment shapes, with few exceptions. These exceptions are rejected on validation.

```toml
[inference.vllm]
model = "PrimeIntellect/INTELLECT-3"
...
```

We will now walk through the supported features and deployment shapes in detail, starting with the single-node deployment.

## Single-Node

The single-node deployment is the default deployment shape. It runs the inference server on a single node. It is useful for debugging, small scale experiments or smaller models. You can configure the single-node deployment with the `SingleNodeInferenceDeploymentConfig` config-field.

```toml
[inference.deployment]
type = "single_node"
```

The launcher starts a `vllm-router` on `inference.server.port` (default `8000`) fronting the vLLM engine on `inference.backend_port` (default `8100`). Clients connect to the router URL; admin operations (weight updates, health checks) bypass the router and hit the engine port directly — the RL entrypoint wires `orchestrator.model.client.admin_base_url` accordingly.

This deployment shape runs the inference server on a single node, if configured with NVLink enabled, it allows you more freedom in terms of parallelism configurations.

```toml
[inference.vllm]
enable_expert_parallel = true # defaults to False
tensor_parallel_size = 2
data_parallel_size = 4

[inference.deployment]
type = "single_node"
```

We reccomend choosing your parallelism based on the expected throughput and latency requirements. High `dp` might create high latency, however it will also give you the highest throughput. This is a tradeoff you need to make based on your use case and required concurrency (see [Adaptive Concurrency](#adaptive-concurrency)). Setting `tp` to a higher value will usually give you lower latency, but the inference server also will become saturated faster with lower number of requests.

Another thing to consider, is the memory usage. You need to make sure that the model will fit into the available GPU memory. We will not go into the details on how to do this in this document. Related thing to consider, is the space for the KV cache. This will heavily affect the amount of requests your inference server can handle. You want to shard your model, either using `inference.vllm.enable_expert_parallel` or `inference.vllm.tensor_parallel_size` to maximize the available GPU memory.

You can also increase the available KV cache memory by enabling `inference.kv_cache_offload`. More details in the [Advanced Configuration](#advanced-configuration) section.


## Multi-Node

This deployment shape branches into 2 sub-shapes:

- [Multi-replica](#multi-replica) - Runs the inference server on multiple nodes, but each node runs an independent vLLM replica. You can think of this as a for-loop over single-node deployments.
- [Wide-EP](#wide-ep) - This option is gated behind `inference.vllm.enable_expert_parallel = true`. It allows you to run the inference server on multiple nodes, allowing you to use multi-node expert parallelism. This is a more advanced feature that is suitable for high-throughput, high-concurrency workloads.

### Multi-replica

This deployment shape runs the inference server on multiple nodes, but each node runs an independent vLLM replica.
Parallelism configuration is the same as the single-node deployment. The shape is defined by setting `inference.deployment.type = "multi_node"` and `inference.deployment.num_nodes` to the number of nodes you want to run the inference server on.

```toml
[inference.deployment]
type = "multi_node"
num_nodes = 2

[inference.vllm]
model = "PrimeIntellect/INTELLECT-3"
tensor_parallel_size = 2
data_parallel_size = 4
```

This configuration will run 2 independent vLLM replicas, each with `tensor_parallel_size=2` and `data_parallel_size=4`. Routing is handled by a single global router running on the first inference node, fronting the per-rank endpoints of all replicas — either `vllm-router` (default) or the upstream `llm-d` EPP+Envoy, selected via the `[inference.router]` block. You can read more about the supported routing options in the [router](#router) section.

### Wide-EP

For huge, 200B+ scale models, you might want to use multi-node expert parallelism to maximize the KV-cache space. This deployment shape is defined by setting `inference.deployment.type = "multi_node"` and `inference.vllm.enable_expert_parallel = true`.

```toml
[inference.deployment]
type = "multi_node"
num_nodes = 2

[inference.vllm]
model = "PrimeIntellect/INTELLECT-3"
enable_expert_parallel = true
tensor_parallel_size = 2
data_parallel_size = 8
```

This configuration will run 2 vLLM processes, each with `data_parallel_size_local = 4` and `tensor_parallel_size = 2` and expert parallelism spanning 2 nodes. The requests are again routed to these processes via the `vllm-router`.

## P/D Disaggregation

This is the most advanced deployment shape. It allows you to disaggregate the prefill and decode stages, with KV cache flowing between them. This is useful for large scale deployments, where there are high requirements on latency, such as agentic workflows spanning 100s of turns.

This deployment shape is defined by setting `inference.deployment.type = "disaggregated"` and choosing how many nodes each prefill and decode replica spans.

```toml
[inference.deployment]
type = "disaggregated"
prefill_nodes_per_replica = 2
decode_nodes_per_replica = 2
```

Sometimes, you may want to run multiple independent vLLM instances within the prefill and decode stages. You can do this by setting `inference.deployment.num_prefill_replicas` and `inference.deployment.num_decode_replicas` to the number of role replicas you want to run.

```toml
[inference.deployment]
type = "disaggregated"
prefill_nodes_per_replica = 2
num_prefill_replicas = 2
decode_nodes_per_replica = 2
num_decode_replicas = 1
```

Now each prefill replica spans 2 nodes and each decode replica spans 2 nodes. With 2 prefill replicas and 1 decode replica, one inference island spans 6 nodes.

For RL runs, the top-level deployment can multiply that whole inference island by setting `deployment.num_infer_replicas`. `deployment.num_infer_nodes` is inferred from the nested inference deployment when you omit it.

```toml
[deployment] # this is a top-level RL deployment, not inference.deployment!!
type = "multi_node"
num_train_nodes = 4

num_infer_replicas = 3
```

This will run 3 inference islands, each running on 6 nodes. The total inference deployment will span 18 nodes, fronted by the single global router.


## Router

Every deployment fronts its vLLM engines with a single global router — it listens on `inference.server.port` and is the one URL clients connect to. The backend is configured via a discriminated `[inference.router]` block (`type = "vllm-router" | "llm-d"`):

```toml
[inference.router]              # or [router] for the standalone inference entrypoint
type = "llm-d"                  # "vllm-router" (default) or "llm-d"
non_cached_tokens = 16          # llm-d only: below this many non-cached prompt tokens, skip remote prefill (P/D)

# llm-d only: base scorer weights, applied to every profile
[inference.router.scorers]
"prefix-cache-scorer" = 3.0
"active-request-scorer" = 2.0

# llm-d only: merged onto the P/D prefill profile (decode_scorer_overrides for decode)
[inference.router.prefill_scorer_overrides]
"queue-scorer" = 2.0
"kv-cache-utilization-scorer" = 2.0
```

- **`vllm-router`** (default) — our fork of [vllm-router](https://github.com/PrimeIntellect-ai/router). Knob: `policy`. The only backend supported for single-node (local) deployments.
- **`llm-d`** — the upstream [llm-d](https://llm-d.ai) Endpoint Picker (EPP) + Envoy proxy (multi-node / disaggregated SLURM deployments only). Routing combines **prefix-cache affinity** (grouped rollouts reuse a cached prefix and skip prefill) with the **`active-request-scorer`** — an in-flight load balancer that spreads requests across ranks immediately, unlike the metrics-scraped `queue-scorer` / `kv-cache-utilization-scorer` / `load-aware-scorer` (which lag and concentrate bursts of same-prefix requests). The scorer weights follow the upstream llm-d P/D guide; tune via `scorers` (base) + `prefill_scorer_overrides` / `decode_scorer_overrides` (per-profile, P/D). Does not support `enable_return_routed_experts` (router replay).

Both backends support the 2 most important things:
- Request routing - KV cache re-use and balanced routing
- P/D disaggregation - handling the prefill and decode stages separately

### Routing policies
The 2 policies you might want to configure are:
- `consistent_hash` - this is the default policy that optimizes for KV cache re-use across turns - it hashes the `X-Session-ID` request header (sent per rollout by the verifiers clients) to pick a replica.

- `round_robin` - this policy will round-robin the requests between the available replicas. This is useful if you want to balance the load between the replicas. This might give you better results if you don't have enough rollouts to make `consistent_hash` hashing saturated.


## Adaptive Concurrency

The orchestrator continuously sizes how many episodes run against the inference engines at once. A static cap cannot be right: too low starves the engines, too high crosses into KV thrash — the engines evict prefix cache, re-prefill the evicted work, and throughput collapses. The optimal value depends on the model, the deployment, and the workload's episode lengths, and it moves as training changes the policy.

The controller's idea is to treat the engines as ground truth and probe, instead of modeling episode cost: while the engines show headroom, admit a little more; when they show pressure, back off. Three behaviors, in increasing severity:

- **Grow**: while the engines look healthy and the current cap is fully used, raise it multiplicatively. Growth is clocked by the pipeline itself — the cap rises by a fixed factor each time the in-flight pool has turned over once — so a fast single-turn workload ramps in seconds while a slow agentic workload ramps at its own pace, with no tuning per workload.
- **Trim**: episodes grow their context *while running*, so a pool that fit comfortably an hour ago can outgrow memory without a single new admission. When KV usage nears the ceiling, the controller first lowers the cap and lets completions drain the pool naturally (soft — no work lost); only if usage keeps climbing despite the closed gate does it also cancel the youngest episodes (hard — least work lost).
- **Cut**: when the engines are demonstrably overloaded — requests being preempted, or piling up in the waiting queue — cut the pool hard, cancel the excess, and hold further cuts until the system settles.

The cap starts from a safe bound derived from the engines (KV capacity divided by the maximum context length), or from `initial_inflight` when a good value is known, and always stays within `[min_inflight, max_inflight]`.

## Advanced Configuration

### KV Cache Offload

Maximizing KV-Cache space is crucial to support high-concurrency workloads. You can offload the KV cache to CPU memory (and, behind it, disk) by setting `inference.kv_cache_offload`. It is a discriminated config with two composable tiers, `cpu` and `disk`: a `cpu` tier is always required, and an optional `disk` tier is layered behind it (GPU → DRAM → disk). Disk-only is not supported.

The `type` field selects the backend:

- `native` — vLLM's built-in offloading. CPU-only uses `OffloadingConnector`; CPU+disk uses `TieringOffloadingSpec` (a CPU primary tier with a filesystem secondary tier). Fully self-contained — no extra processes.
- `mooncake` — a [Mooncake](https://github.com/kvcache-ai/Mooncake) **shared distributed store** (SLURM only). One `mooncake_master` + metadata server runs on the head inference node; every inference node runs a `mooncake_client` that contributes its DRAM (and, with `disk`, SSD) segment to that *single* pool. Because blocks are keyed by model + parallel rank + content hash (no instance id), a prefix cached by one node/replica is reusable by all of them over RDMA — pooling every node's CPU RAM into one KV cache. Use `native` for local/single-process runs.

```toml
# Native CPU offload (reserves 128GB of CPU KV cache for this instance)
[inference.kv_cache_offload]
type = "native"
[inference.kv_cache_offload.cpu]
num_bytes = 128_000_000_000 # 128GB

# Native CPU + disk tiering (self-contained)
[inference.kv_cache_offload]
type = "native"
[inference.kv_cache_offload.cpu]
num_bytes = 128_000_000_000
[inference.kv_cache_offload.disk]
path = "/scratch/kv"        # disk capacity is bounded by the filesystem

# Mooncake CPU + disk (per-node distributed store, RDMA)
[inference.kv_cache_offload]
type = "mooncake"
[inference.kv_cache_offload.cpu]
num_bytes = 128_000_000_000
[inference.kv_cache_offload.disk]
path = "/scratch/kv"
```

For `native`, `cpu.num_bytes` is the aggregate CPU KV pool for the instance (vLLM shards it across workers). For `mooncake`, `cpu.num_bytes` is the DRAM each node contributes to the shared pool (so the total pool ≈ `num_bytes × #inference-nodes`); the store uses RDMA, so it requires an RDMA-capable fabric. Enabling offload automatically enables prefix caching.


### Optimized P/D disaggregation deployment

For optimal P/D disaggregation deployment, we automatically set the decode `all2all_backend` to `deepep_low_latency` and the prefill `all2all_backend` to `deepep_high_throughput`. We currently don't support customizing all2all backends for P/D disaggragation out of the box. You can do this by overriding the slurm template only.

For KV cache transfer, we utilize the NIXL connector. This is the default and only currently supported connector. We aim to support more advanced options, such as D->P transfer, or Mooncake Connector in the future.

> **Required:** The pip-wheel NIXL's bundled UCX segfaults on the prefill→decode KV transfer. You must build NIXL against UCX 1.19.x from source — see [Disaggregated Prefill/Decode Inference](advanced.md#disaggregated-prefilldecode-inference) in the Advanced docs for the full setup.

For configuring various knobs with environment variables, we enable you to configure prefill and decode environment variables separately. This is useful if you want to configure different environment variables for the prefill and decode stages.

```toml
[inference.deployment]
type = "disaggregated"

[inference.deployment.prefill_env_vars]
"VLLM_ENABLE_MOE_DP_CHUNK" = "0"
"VLLM_DEEP_GEMM_WARMUP" = "skip"

[inference.deployment.decode_env_vars]
"VLLM_DEEP_GEMM_WARMUP" = "skip"
```

These are role-specific and layer on top of [`env_vars`](configuration.md#environment-variables) shared by all inference processes regardless of role.

### Other vLLM features
The `[inference.vllm]` section is a pass-through: every key is forwarded to the vLLM server under vLLM's own argument name, whether prime-rl types it or not. Anything `vllm serve` accepts can be set here.

```toml
[inference.vllm]
headless = true
max_num_seqs = 256
```

On the CLI the same keys are available as `--inference.vllm.max-num-seqs 256` (or `--vllm.max-num-seqs 256` for the standalone inference entrypoint); dict-valued arguments take a JSON string, e.g. `--vllm.compilation-config '{"cudagraph_mode": "NONE"}'`.

### Router Replay

Router replay works by capturing the expert routing decisions into a buffer. This buffer then gets sent to the trainer, which can use it instead of re-computing the routing. This lowers the trainer↔inference mismatch by an order of magnitude, resulting in more stable training.

To enable router replay, you can set `inference.vllm.enable_return_routed_experts = true`.

```toml
[trainer]
enable_router_replay = true # this will also auto-set inference.vllm.enable_return_routed_experts = true

[inference.vllm]
enable_return_routed_experts = true
```

This however is not free, it adds a significant overhead to the HTTP requests as this payload can grow quite large. We reccomend sizing up the env server pool (`orchestrator.*.source.serve.pool`) to allow for more parallelization on the verifiers side.

Currently this feature is also not supported with CPU KV cache offload, which can have negative impact on the inference throughput.

### Sampling Replay

Truncated sampling (`top_p < 1`, `top_k`) renormalizes the sampling distribution over the surviving "kept set" of tokens. The rollout logprobs reflect that (`logprobs_mode = "processed_logprobs"`), so the trainer must renormalize over the same set — otherwise every importance ratio is biased and training collapses (DeepSeek V3.2's "Keep Sampling Mask", [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) §3.1; Cognition's [SWE-1.7 post](https://cognition.com/blog/swe-1-7)). prime-rl handles this automatically: vLLM records the kept-set token ids at sampling time (`--return-sampling-mask`, native since vLLM 0.28) and the trainer renormalizes its logprobs over them.

```toml
[orchestrator.train.sampling]
top_p = 0.95
top_k = 512   # optional, defaults to 512 under truncation (bounds the kept sets)
```

That's all — there are no replay flags. Truncated train sampling makes the inference server return kept sets (`inference.enable_return_sampling_mask`, auto-enabled) and the trainer replays whatever masks arrive. Train-sampling `top_k` above 512 is rejected: the trainer pads each micro batch's masks to the largest kept set, so the bound caps trainer memory. Configs that would break under renormalized logprobs are rejected: `opd`/`opsd` and truncation knobs smuggled via `extra_body`; Gemma-family (softcapped) lm_heads fail loudly. Frozen-source envs are exempt.

Capture runs on vLLM's V2 model runner (forced automatically) and is engine-wide, which has two consequences for other traffic on the same server. It is incompatible with router replay (`enable_return_routed_experts`, V1-only). And while it is on, vLLM rejects any request with `temperature <= 0` or without an effective `top_k > 0` — eval sampling against the training server must set `top_k` (the model's generation config often supplies one) and a non-zero temperature.

When launching the inference server standalone, set `inference.enable_return_sampling_mask = true` yourself; clients must sample with `0 < top_k <= 512`.
