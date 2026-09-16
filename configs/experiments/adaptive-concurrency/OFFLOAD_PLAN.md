# KV offload experiment

## Comparison and decision gate

Baseline code: merge `936d7ef4f` of main `d7339fcf2` (linear GPU-KV growth scheduler). Initial config commits: `585f5c7c3`, `414c623af`; selected Mooncake/setup revision: `663bf57f9`. Controller equals main; no offload-aware controller change is present.

| Run | Job | Layout | CPU KV |
| --- | --- | --- | --- |
| terminal-lego-no-offload-v2 | pending submission | 1 trainer + 1 TP8/EP inference node | none |
| terminal-lego-mooncake-dram-v1 | pending submission | same | 1.2 TB Mooncake DRAM |

Both stack `base.toml` with `no-offload.toml` or `offload.toml`: Qwen3-30B-A3B-Thinking-2507, Terminal Lego, 64k context, group 8, batch 128, 50 steps, max inflight 2048. Fixed default source/inference seeds match; asynchronous completion and sampling mean the workloads will not be identical. The trainer settings are unchanged. Attempt 681 (native-offload-v1, 512 GB) was cancelled during startup before training; exclude it from results. Prior TP8 runs had roughly 984 GB aggregate GPU KV; native eager CPU offload should exceed that to avoid just mirroring the GPU cache. The 1.2 TB tier leaves roughly 400 GB host headroom on these nodes.

Do not launch the modified-controller run until the baseline comparison establishes whether offload is worse, where it loses time, and whether a controller mechanism explains it. If the expected ordering is absent, report that rather than manufacturing it. Use the same offload config for the third run with only run identifiers and the controller revision changed.

## Backend inventory (installed vLLM 0.28.0)

| Backend | PRL support today | Useful facts |
| --- | --- | --- |
| OffloadingConnector / CPUOffloadingSpec | First-class `inference.kv_cache_offload.type = "native"` plus CPU bytes | Self-contained async GPU/CPU copies, eager completed prompt blocks, CPU LRU eviction. Initially investigated; shared-memory mount blocks a full-size baseline here. |
| OffloadingConnector / TieringOffloadingSpec | First-class native CPU + disk | CPU staging plus filesystem disk. Main includes the `fs` tier-name fix. vLLM also offers object/P2P/custom secondary tiers, but these lack PRL convenience configuration/orchestration. |
| MooncakeStoreConnector | First-class `type = "mooncake"`, SLURM launcher services | Selected baseline: shared DRAM pool, no SSD. Backend RPC metrics plus separate store metrics; verify successful saving and eviction under sustained pressure. |
| SimpleCPUOffloadConnector | vLLM connector pass-through, no PRL typed backend | Installed; eager/lazy CPU and disk modes. No connector Prometheus stats implementation found in this installed backend. Validate separately before adopting. |
| LMCacheConnectorV1 / LMCacheMPConnector | vLLM pass-through; external package/service/config needed | CPU, disk and remote cache ecosystem. Not a turnkey PRL deployment. |
| FlexKVConnectorV1 and other external connectors | vLLM pass-through; external runtime dependencies needed | Integration availability is not deployment validation. |
| Weight offloading (UVA/prefetch) | vLLM arguments pass through | Moves model parameters, not KV; different tradeoff and outside this KV-cache comparison. |

When using connector pass-through, leave PRL `kv_cache_offload` unset; otherwise PRL constructs its connector configuration. Native and Mooncake can also compose with NIXL for P/D, but these experiments use combined prefill/decode.

## What the metrics mean

PRL already scrapes arbitrary `vllm:` gauges, counters and histograms to engine/aggregate metrics. The controller only receives GPU usage, KV capacity, running/waiting requests, capacity waiting, and preemption deltas. Other telemetry is logged but not used in decisions.

- GPU `kv_cache_usage_perc`: non-free blocks. Evictable prefix blocks count as free. This is active/pinned GPU pressure, not physical allocation or all cached prefixes.
- Native `kv_offload_cpu_cache_usage_perc`: CPU blocks pinned by in-flight transfers, excluding evictable cached blocks. `...write_usage_perc` and `...read_usage_perc` split it. Full reusable CPU LRU occupancy is normal and must not itself trigger trimming.
- Native `kv_offload_allocation_failure_total`: failed store allocation attempts; correlate deltas with pinned fraction and requested allocation sizes. One skipped save is not grounds to cancel a rollout.
- Native `kv_offload_load_bytes_total`, `...store_bytes_total`, `...load_time_total`, `...store_time_total`, and size histograms: transfer volume and duration. Summed rank/job durations can overlap; do not mistake time/wall-time for a normalized utilization percentage.
- `kv_offload_lookup_sync_delay_seconds` / `...async_delay_seconds`: lookup latency histograms. CPU saturation is only one possible bottleneck; bandwidth and scheduling can bottleneck with low pinned fraction.
- `num_requests_waiting_by_reason{reason="capacity"|"deferred"}`: capacity admission versus transient constraints (including KV transfers). Deferred is not exclusively offload. PRL retains this label; other connector labels such as Mooncake operation/status are currently summed away, so inspect raw Prometheus for per-operation diagnosis.
- Local `prefix_cache_hits_total/queries_total` and external `external_prefix_cache_hits_total/queries_total` must be read separately. External queries cover the externally checked token subset; do not add local and external hit percentages. Combined avoided-work ratio needs a common denominator and verified token accounting.
- Generation TPS, prompt compute work, TTFT, inter-token latency, episode completions per minute, effective training tokens/s, errors/cancellations and trainer waiting quantify useful performance. Raw prompt-token TPS may count cached work and can mislead.
- PRL salts requests by the policy version at episode dispatch. Reuse is partitioned by policy; more cache cannot create cross-policy hits. Do not change cache versioning as a concurrency optimization.

## Minimal controller candidates, in order

1. Retain GPU bootstrap capacity and all existing GPU safety signals. Do not sum GPU and CPU capacities: active attention still requires GPU residency, data can be duplicated, and transfer bandwidth is a separate bound.
2. Check whether persistent deferred waiting is the dominant growth freeze despite low GPU pressure, low transfer occupancy, and healthy throughput. If confirmed, base the general queue growth gate on capacity waiting (with total waiting fallback for older engines), and retain an independent offload pressure guard. Do not treat every normal async load as overload.
3. If native transfer occupancy actually approaches saturation, use one optional normalized offload pressure field. Reuse the linear growth schedule against the worst pressure, and soft-trim on persistent offload pressure/allocation failures. GPU hard cuts remain emergency safeguards; CPU cache pressure alone should drain naturally, not cancel useful work. Thresholds must follow baseline data; blindly applying GPU thresholds to CPU transfer occupancy is uncalibrated.
4. If TPS stalls while GPU and pinned CPU pressure remain low, occupancy is insufficient. Investigate queue/lookup latency and transfer service time before adding a small throughput-aware growth hold. Avoid a multi-parameter PID or fabricated capacity model. If backend overhead or cache policy dominates, a controller change cannot promise to beat no-offload.

## Validation

Analyze ramp and sustained operation separately, with time-weighted metrics, same trainer step ranges and comparable turnover/context distributions. Report cap/actual inflight, time spent growth-gated by each reason, hard cut count and cancelled work, GPU KV p50/p95/max, capacity/deferred queue, CPU transfer occupancy, save failures, external loads/hits, inference TPS and effective training tokens per wall-clock second. Compare both full runs and post-ramp windows; if 50 steps do not establish steady behavior, explicitly mark the result inconclusive before changing run length.

Success: no hard controller cuts or preemptions, smooth admitted concurrency, stable GPU pressure below the hard threshold, productive external hits with manageable transfer cost, and improved useful throughput. The offloaded fixed run beating the no-offload baseline is a hypothesis, not a guaranteed outcome.

## Sources

- [vLLM 0.28 KV offloading guide](https://docs.vllm.ai/en/v0.28.0/features/kv_offloading_usage/) (tier architecture, sizing, default prompt-only stores).
- [CPU spec and gauge definitions](https://docs.vllm.ai/en/v0.28.0/api/vllm/v1/kv_offload/cpu/spec/).
- [Mooncake store guide](https://docs.vllm.ai/en/v0.28.0/features/mooncake_store_connector_usage/).
- [LMCache connector](https://docs.vllm.ai/en/v0.28.0/api/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector/).
- [FlexKV connector](https://docs.vllm.ai/en/v0.28.0/api/vllm/distributed/kv_transfer/kv_connector/v1/flexkv_connector/).
- Installed source inspected: `vllm/v1/core/block_pool.py`, `vllm/v1/kv_offload/cpu/{spec,manager,common}.py`, connector offloading metrics/scheduler, `simple_cpu_offload_connector.py`, and `vllm/v1/metrics/loggers.py`.
- PRL source inspected: config `inference.py`, `orchestrator/{concurrency,inference_metrics,dispatcher,envs}.py`, inference server and launcher templates.

## Infrastructure finding — decision pending

On the allocated H200 node, `free -h` reports 1.5 TiB RAM but `/dev/shm` is mounted at 512 GiB (484 GiB free). Native vLLM 0.28 uses `/dev/shm/vllm_offload_{engine_id}.mmap` and checks the full allocation against shared-memory space; there is no configured alternate path in the installed implementation. Job 682 was cancelled during startup before a known allocation failure. Passwordless sudo is unavailable. The 1.2 TB config is valid but cannot run on this mount. User asked to choose between a healthy Mooncake DRAM store with full GPU memory, matched smaller GPU KV budgets for native, or arranging a larger shared-memory mount. No backend/controller result can be inferred from these startup attempts.

## Selected Mooncake setup

User selected Mooncake after the native shared-memory limitation was confirmed. CPU DRAM segment is 1.2 TB and the full GPU KV budget is retained. Mooncake `enable_offload` is false: that flag controls SSD offload, not GPU-to-DRAM caching. The existing one-hour read lease is reduced to 60 seconds, which must exceed observed lookup-to-load latency while permitting idle prefixes to be evicted. Verify invalid/expired KV load counters remain zero. This is a prerequisite store correction, applied before the controller comparison, not evidence of a controller improvement.

Installed Mooncake 0.3.11.post1 defaults are 95% memory eviction high watermark and 5% eviction fraction. Native CPU transfer-pinned gauges are not available for Mooncake; do not invent an analogous utilization from total store occupancy. Use raw `vllm:mooncake_store_operation_{total,keys_total,bytes_total,failed_keys_total,time_seconds}` by operation/status plus master allocation/eviction metrics on port 9003. A full evictable cache can be healthy. Failed stores or growing load latency provide more actionable feedback than occupied bytes alone.

The first no-offload attempt (680) was healthy at inference but launched no environment server due to an upstream RL template mismatch (`env_names` passed, `train_env_names`/`eval_env_names` read). It was cancelled before training. Corrected template uses `env_names.train` / `env_names.eval`. Both new baselines run this same launcher correction, and generated scripts were checked for environment launch and Bash syntax. No controller code differs from main.

For the Mooncake controller candidate, first quantify persistent *deferred* waiting versus *capacity* waiting and correlate it with successful load completion and TPS. Only if demonstrated, narrow the growth gate to capacity waiting while adding minimal soft backpressure for persistent store failures or transfer delay. Preserve GPU safety cuts. A transfer wait alone does not establish saturation; a repeatedly failing store can be degraded despite healthy GPU occupancy. Baseline metrics determine which, if any, change is justified.

[Mooncake store lease/eviction design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/store/mooncake-store.md#lease) explains why hour-long read leases can obstruct eviction. [vLLM Mooncake configuration](https://docs.vllm.ai/en/v0.28.0/features/mooncake_store_connector_usage/) distinguishes DRAM segments from SSD offload.
