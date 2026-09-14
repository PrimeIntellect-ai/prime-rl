# Adaptive concurrency baselines

These runs measure the default adaptive concurrency controller on ScaleSWE.
Each run uses 20 training steps and three H200 nodes.

| Overlay | Model | Inference setup |
| --- | --- | --- |
| `pd-disagg.toml` | Qwen3-30B-A3B-Thinking-2507 | One prefill node and one decode node |
| `mooncake.toml` | Qwen3-30B-A3B-Thinking-2507 | Two replicas with a shared Mooncake cache |
| `linear-attention.toml` | Qwen3.5-35B-A3B | Two replicas with hybrid linear attention |

All runs use one trainer node, a 65,536-token context, full CPU offload,
and SignSGD. The controller uses its default bounds and derives its initial
limit from live inference metrics.

Stack the shared settings with one experiment overlay:

```bash
uv run rl @ configs/experiments/adaptive-concurrency/base.toml \
  @ configs/experiments/adaptive-concurrency/pd-disagg.toml
```

## Metrics

Use the local `monitors/file/metrics.jsonl` file as the source of record.
W&B receives the same metrics.

Summarize one or more completed runs from the repository root:

```bash
uv run python tools/analyze_adaptive_concurrency.py <run-dir> [<run-dir> ...]
```

| Question | Metric |
| --- | --- |
| Is the concurrency cap smooth? | `concurrency/max_inflight` |
| Does the active pool follow the cap? | `dispatcher/inflight/train` |
| Does KV pressure cause cache thrash? | `inference/agg/num_preemptions_total:rate/sum` |
| Are requests waiting for capacity? | `inference/agg/num_requests_waiting_reason_capacity/sum` |
| Are requests waiting for any reason? | `inference/agg/num_requests_waiting/sum` |
| What is inference throughput? | `inference/agg/generation_tokens_total:rate/sum` |
| Is GPU KV usage near its limit? | `inference/agg/kv_cache_usage_perc/max` |
| Does prefix reuse collapse? | `inference/agg/prefix_cache_hit_rate/min` |

For PD disaggregation, also compare the `inference/prefill/*` and
`inference/decode/*` scopes. For Mooncake, inspect the external prefix-cache
hit and query counters. For Qwen3.5, compare throughput and request queues
against KV usage because linear-attention state can change their relationship.

## Evaluation

Ignore startup samples before the first nonzero generation throughput sample.
Plot all wall-time inference samples and overlay the step-based controller
gauges. Check these behaviors:

1. The cap grows gradually while the active pool is binding and engines are clear.
2. A persistent queue, preemption, or high KV usage causes one controlled reduction.
3. The cap settles instead of alternating between growth and sharp reductions.
4. Generation throughput stays near its best sustained level after the cap settles.
5. No single engine hides a pressure event inside a healthy fleet average.

Use the baseline evidence to change one controller behavior at a time. Repeat
the affected baseline with the same model, context length, workload, and node
layout.
