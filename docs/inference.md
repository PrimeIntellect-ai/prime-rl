# Inference

This page covers the inference configuration and the supported features/deployment shapes. It covers how to scale the inference server from a single GPU to 1000s of GPUs that run agentic workloads at the speed of light with all the bells and whistles configured.

## Table of Contents

- [Overview](#overview)
- [Single-Node](#single-node)
- [Multi-Node](#multi-node)
    - [Multi-replica](#multi-replica)
    - [Wide-EP](#wide-ep)
- [P/D Dissagregation](#pd-dissagregation)
- [Router](#router)
    - [Routing policies](#routing-policies)
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
- [Disaggregated](#disaggregated) - Runs the inference server on multiple nodes, but disaggregates the prefill and decode stages. Useful for large scale experiments or larger models, where latency is a concern and multi-node deployment creates very high E2E rollout latency, such as agentic workflows.

Most of the features are supported for all deployment shapes, with few exceptions. These exceptions are rejected on validation.

You can select the deployment shape with `InferenceDeploymentConfig` in your config file. This is a config-field that allows you to set the deployment shape, deployment-specific knobs such as `num_nodes`, `num_replicas`, `backend_port`, and a `[...deployment.router]` block, etc.

```toml
[inference.deployment]
type = "single_node" # or "multi_node" or "disaggregated"
```

To configure the inference server, you can use the `InferenceConfig` field. This is a config-field that allows you to set the inference server-specific knobs. Most of these are supported for all deployment shapes, with few exceptions. These exceptions are rejected on validation.

```toml
[inference]
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

This deployment shape runs the inference server on a single node, if configured with NVLink enabled, it allows you more freedom in terms of parallelism configurations.

```toml
[inference]
enable_expert_parallel = true # defaults to False

[inference.parallel]
tp = 2
dp = 4

[inference.deployment]
type = "single_node"
```

We reccomend choosing your parallelism based on the expected throughput and latency requirements. High `dp` might create high latency, however it will also give you the highest throughput. This is a tradeoff you need to make based on your use case and required `orchestrator.max_inflight_requests`. Setting `tp` to a higher value will usually give you lower latency, but the inference server also will become saturated faster with lower number of requests.

Another thing to consider, is the memory usage. You need to make sure that the model will fit into the available GPU memory. We will not go into the details on how to do this in this document. Related thing to consider, is the space for the KV cache. This will heavily affect the amount of requests your inference server can handle. You want to shard your model, either using `inference.enable_expert_parallel` or `inference.parallel.tp` to maximize the available GPU memory.

You can also increase the available KV cache memory by enabling `inference.kv_cache_offload`. More details in the [Advanced Configuration](#advanced-configuration) section.


## Multi-Node

This deployment shape branches into 2 sub-shapes:

- [Multi-replica](#multi-replica) - Runs the inference server on multiple nodes, but each node runs an independent vLLM replica. You can think of this as a for-loop over single-node deployments.
- [Wide-EP](#wide-ep) - This option is gated behind `inference.enable_expert_parallel = true`. It allows you to run the inference server on multiple nodes, allowing you to use multi-node expert parallelism. This is a more advanced feature that is suitable for high-throughput, high-concurrency workloads.

### Multi-replica

This deployment shape runs the inference server on multiple nodes, but each node runs an independent vLLM replica.
Parallelism configuration is the same as the single-node deployment. The shape is defined by setting `inference.deployment.type = "multi_node"` and `inference.deployment.num_nodes` to the number of nodes you want to run the inference server on.

```toml
[inference.deployment]
type = "multi_node"
num_nodes = 2

[inference]
model = "PrimeIntellect/INTELLECT-3"

[inference.parallel]
tp = 2
dp = 4
```

This configuration will run 2 independent vLLM replicas, each with `tp=2` and `dp=4`. Routing is handled by a router instance running on the same node as the 1st replica — either `vllm-router` (default) or the upstream `llm-d` EPP+Envoy, selected via the `[...deployment.router]` block. You can read more about the supported routing options in the [router](#router) section.

### Wide-EP

For huge, 200B+ scale models, you might want to use multi-node expert parallelism to maximize the KV-cache space. This deployment shape is defined by setting `inference.deployment.type = "multi_node"` and `inference.enable_expert_parallel = true`.

```toml
[inference.deployment]
type = "multi_node"
num_nodes = 2

[inference]
model = "PrimeIntellect/INTELLECT-3"
enable_expert_parallel = true

[inference.parallel]
tp = 2
dp = 8
```

This configuration will run 2 vLLM processes, each with `data_parallel_size_local = 4` and `tp = 2` and expert parallelism spanning 2 nodes. The requests are again routed to these processes via the `vllm-router`.

## P/D Dissagregation

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

This will run 3 inference islands, each running on 6 nodes. The total inference deployment will span 18 nodes and start 3 separate router instances.


## Router

Multi-node and disaggregated deployments front their vLLM backends with a router, configured via a discriminated `[...deployment.router]` block (`type = "vllm-router" | "llm-d"`):

```toml
[inference.deployment.router]   # or [deployment.router] for the standalone inference entrypoint
type = "llm-d"                  # "vllm-router" (default) or "llm-d"
# llm-d-only knobs (all optional):
scorers = { "prefix-cache-scorer" = 3.0, "active-request-scorer" = 2.0 }   # base, applied to every profile
prefill_scorer_overrides = { "queue-scorer" = 2.0, "kv-cache-utilization-scorer" = 2.0 }  # merged onto the P/D prefill profile
decode_scorer_overrides = {}    # merged onto the P/D decode profile
non_cached_tokens = 16          # below this many non-cached prompt tokens, skip remote prefill (P/D)
```

- **`vllm-router`** (default) — our fork of [vllm-router](https://github.com/PrimeIntellect-ai/router). Knob: `policy`.
- **`llm-d`** — the upstream [llm-d](https://llm-d.ai) Endpoint Picker (EPP) + Envoy proxy. Routing combines **prefix-cache affinity** (grouped rollouts reuse a cached prefix and skip prefill) with the **`active-request-scorer`** — an in-flight load balancer that spreads requests across ranks immediately, unlike the metrics-scraped `queue-scorer` / `kv-cache-utilization-scorer` / `load-aware-scorer` (which lag and concentrate bursts of same-prefix requests). The scorer weights follow the upstream llm-d P/D guide; tune via `scorers` (base) + `prefill_scorer_overrides` / `decode_scorer_overrides` (per-profile, P/D). Does not support `enable_return_routed_experts` (router replay).

Both backends support the 2 most important things:
- Request routing - KV cache re-use and balanced routing
- P/D disaggregation - handling the prefill and decode stages separately

### Routing policies
The 2 policies you might want to configure are:
- `consistent_hash` - this is the default policy that optimizes for KV cache re-use across turns - this works by hashing a request header to determine where to route the request to. You can configure what to hash by setting
`orchestrator.model.client.extra_headers_from_state` to the header the `router` expects to be set.

We set it to a sensible default, that works with all verifiers environments.

```toml
[orchestrator.model.client.extra_headers_from_state]
X-Session-ID = "trajectory_id" # this is the default - each rollout has a unique trajectory_id and router expects X-Session-ID
```

- `round_robin` - this policy will round-robin the requests between the available replicas. This is useful if you want to balance the load between the replicas. This might give you better results if you don't have enough rollouts to make `consistent_hash` hashing saturated.


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

prefill_env_vars = {"VLLM_ENABLE_MOE_DP_CHUNK"="0", "VLLM_DEEP_GEMM_WARMUP"="skip"}
decode_env_vars = {"VLLM_DEEP_GEMM_WARMUP"="skip"}
```

These are role-specific and layer on top of [`env_vars`](configuration.md#environment-variables) shared by all inference processes regardless of role.

### Other vLLM features
We support various other vLLM features. Some of those, such as `enable_dbo`, `enable_eplb` are exposed as a top-level config fields. For those that are not, you can configure them by setting `inference.vllm_extra` to the desired value.

```toml
[inference.vllm_extra]
headless = true
```

### Router Replay

Router replay works by capturing the expert routing decisions into a buffer. This buffer then gets sent to the trainer, which can use it instead of re-computing the routing. This lowers the trainer↔inference mismatch by an order of magnitude, resulting in more stable training.

To enable router replay, you can set `inference.enable_return_routed_experts = true`.

```toml
[trainer]
enable_router_replay = true # this will also auto-set the inference.enable_return_routed_experts = true

[inference]
enable_return_routed_experts = true
```

This however is not free, it adds a significant overhead to the HTTP requests as this payload can grow quite large. We reccomend increasing `orchestrator.*.env.num_workers` to allow for more parallelization on the verifiers side.

Currently this feature is also not supported with CPU KV cache offload, which can have negative impact on the inference throughput.
