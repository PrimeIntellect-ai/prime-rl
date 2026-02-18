# Expert Parallel Deployment

PRIME-RL exposes vLLM expert-parallel (EP) settings for MoE inference.

## Quick Start (EP Only)

Run the EP-only debug config:

```bash
uv run rl @ configs/debug/rl_ep.toml
```

This enables EP without prefill/decode disaggregation.

## Core EP Config

Set EP in `[inference]`:

- `enable_expert_parallel = true`
- `all2all_backend = "<backend>"`
- `enable_eplb = true|false` (optional load balancing)

Supported `all2all_backend` values:

- `allgather_reducescatter`
- `deepep_high_throughput`
- `deepep_low_latency`
- `flashinfer_all2allv`
- `mori`
- `naive`
- `pplx`

## Backend Selection Guide (Practical)

| Backend | Typical fit |
| --- | --- |
| `allgather_reducescatter` | safest default across mixed workloads |
| `pplx` | single-node EP setups |
| `deepep_high_throughput` | prefill-heavy, throughput-oriented paths |
| `deepep_low_latency` | decode-heavy, low-latency paths |
| `flashinfer_all2allv` | specialized NVLink/MNNVL-focused environments |
| `mori` | ROCm-oriented deployments |
| `naive` | debugging only |

Use vLLM's backend selection guide for prerequisites and tuning:
[Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/).

## EPLB Tuning Notes

vLLM supports richer EPLB tuning through `eplb_config` (for example `window_size`, `step_interval`, `num_redundant_experts`, `log_balancedness`, `use_async`, and `policy`).

In PRIME-RL today:

- `enable_eplb` is exposed directly.
- advanced `eplb_config` fields use vLLM defaults unless PRIME-RL config is extended.

For memory-constrained runs, be conservative with redundant experts because EPLB increases memory footprint.

## EP with PD Disaggregation

Run the combined example:

```bash
uv run rl @ configs/debug/rl_pd_disagg_ep.toml
```

In PD mode, EP is still configured globally in `[inference]`, with optional role-specific backend overrides in `[pd_disagg]`:

- `prefill_all2all_backend`
- `decode_all2all_backend`

This is useful when prefill and decode have different communication profiles.

Recommended starting point:

- prefill: `deepep_high_throughput`
- decode: `deepep_low_latency`
- fallback: `allgather_reducescatter` if prerequisite kernels are unavailable

