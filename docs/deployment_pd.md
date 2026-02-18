# PD Disaggregation Deployment

PRIME-RL supports prefill/decode (PD) disaggregated inference with vLLM.
In PD mode, `uv run rl` starts:

- one or more prefill vLLM workers
- one or more decode vLLM workers
- one local proxy that does prefill (`max_tokens=1`) then decode

## Quick Start

Run the debug config for a local smoke test:

```bash
uv run rl @ configs/debug/rl_pd_disagg.toml
```

## Connector Selection (P2P NCCL vs NIXL)

Use this as a practical starting point:

| Connector | Best fit | Operational profile | Notes |
| --- | --- | --- | --- |
| `P2pNcclConnector` | Local/single-node bring-up, smaller xPyD | Simple, fast to validate | Strong default for early rollout and debugging. |
| `NixlConnector` | Multi-node and stricter production latency goals | More infra/runtime dependencies | Better long-term path for larger distributed deployments. |

In PRIME-RL, choose via:

- `pd_disagg.kv_connector`
- `pd_disagg.kv_send_type` (mainly relevant to P2P NCCL flow)

## Topology and Routing

- **xPyD-style pools:** prefill and decode each support multiple workers.
- **Health-aware routing:** failing workers are temporarily quarantined and requests retry on healthy workers.
- **Static client mode only:** `orchestrator.client.elastic` is currently unsupported in PD mode.

## Worker and Parallelism Mapping

- `pd_disagg.prefill_gpu_ids` and `pd_disagg.decode_gpu_ids` define the role-specific GPU pools.
- Role TP defaults to `inference.parallel.tp`.
- Override per role via `pd_disagg.prefill_tp` and `pd_disagg.decode_tp`.
- Each contiguous TP-sized chunk of GPU IDs is one worker.
- Each worker currently runs with `dp=1` (parallelism scales by worker count, not per-worker vLLM DP).

Example:

- `prefill_gpu_ids = [0,1,2,3]`, `prefill_tp = 2` -> workers `[0,1]` and `[2,3]`
- `decode_gpu_ids = [4,5]`, `decode_tp = 1` -> workers `[4]` and `[5]`

## Host and Port Behavior

- `pd_disagg.auto_ports = true` (default) auto-resolves non-overlapping proxy/server/KV ports at runtime.
- Host defaults to `inference.server.host` (fallback: `127.0.0.1`).
- For deterministic endpoints, set `pd_disagg.auto_ports = false` and explicitly set:
  - `host`
  - `proxy_port`
  - `prefill_port`
  - `decode_port`
  - `prefill_kv_port`
  - `decode_kv_port`

## Weight Broadcast in PD Mode

Both filesystem and NCCL weight broadcast are supported.

NCCL constraints in PD mode:

- `max_async_level = 1`
- LoRA disabled
- matching `pd_disagg.prefill_tp` and `pd_disagg.decode_tp`

## Runtime Caveats

- Keep role TP symmetric (`prefill_tp == decode_tp`) when using NCCL weight broadcast.
- For role-asymmetric EP tuning, prefer backend overrides first (`prefill_all2all_backend`, `decode_all2all_backend`) before changing worker topology.
- Validate connector dependencies in the target environment before large runs (especially for non-default connector stacks).

## Expert Parallel with PD

Use global EP settings in `[inference]`:

- `enable_expert_parallel`
- `all2all_backend`
- `enable_eplb`

Optional role-specific backend overrides in `[pd_disagg]`:

- `prefill_all2all_backend`
- `decode_all2all_backend`

For EP tuning details, see [Expert Parallel Deployment](deployment_ep.md).

