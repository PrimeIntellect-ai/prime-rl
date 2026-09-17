# Sandbox runtime benchmark

Compares the `prime`, `modal` and `e2b` sandbox runtimes side by side on the same
agentic workload. Everything but the runtime is held fixed:

| | |
|---|---|
| Tasksets | Terminal-Bench 2 at avg@4 (89 tasks x 4), SWE-bench Verified at avg@1 (500 tasks), SWE-bench Pro at avg@1 (731 tasks) |
| Harness | `bash` (runs inside the sandbox; the model is reached through the interception tunnel) |
| Model | `deepseek/deepseek-v4.1-flash` on Prime Inference |
| Concurrency | pinned at 1000 in-flight rollouts across the three sources |
| Rollout timeout | 3600 s per rollout |
| Retries | provider errors only, so sandbox failures land as failed episodes and count against the runtime |

Each task's image and resource request come from its Harbor `task.toml`; the runtime
config only names the provider (plus a label on prime and a 2 h sandbox lifetime on e2b).

## Run

One eval per runtime, in this order: prime, then modal, then e2b once its account limits
cover 1000 concurrent sandboxes and 2 h sandbox lifetimes.

```bash
uv run eval @ benchmarks/runtimes/prime.toml --run.name runtime-bench-prime
uv run eval @ benchmarks/runtimes/modal.toml --run.name runtime-bench-modal
uv run eval @ benchmarks/runtimes/e2b.toml   --run.name runtime-bench-e2b
```

Credentials in the eval process' environment: `PRIME_API_KEY` (Prime Inference and prime
sandboxes), a Modal token (`modal token new` or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`),
`E2B_API_KEY`. `--resume` continues an interrupted run from its trace stream.

First-use image costs are part of what this measures, and they differ per provider:

- prime builds a VM image per task image on first use (~10 min) and caches it; the trace's
  `PrimeRuntimeInfo.image_cached` says which rollouts waited for a build.
- e2b builds a template per (image, cpu, memory) on first use (about a minute) and caches it
  by name; the run has ~1300 distinct task images.
- modal pulls the image per sandbox.

## Analyze

```bash
uv run python benchmarks/runtimes/analyze.py outputs/runtime-bench-prime outputs/runtime-bench-modal outputs/runtime-bench-e2b
```

Per run and source the script prints, from the trace stream:

- rollouts, share ok, rollouts that hit a `SandboxError`;
- boot (sandbox provisioning, `timing.boot`), setup (`timing.setup`) and rollout wall clock
  (dispatch to arrival) at p50 / p90; prime rows add the boot split by cached vs built image;
- the resources each sandbox requested (`agent.runtime` on the trace);
- the summed sandbox lifetime, counted from the sandbox being up to the end of scoring;
- the cost that lifetime implies at the provider's published per-resource rates.

| provider | per core-hour | per GB-hour | per GB disk-hour | source |
|---|---:|---:|---:|---|
| prime | $0.05 | $0.01 | $0.001 | [docs.primeintellect.ai/sandboxes/overview](https://docs.primeintellect.ai/sandboxes/overview) |
| modal | $0.1419 | $0.0240 | - | [modal.com/pricing](https://modal.com/pricing) |
| e2b | $0.0504 | $0.0162 | - | [e2b.dev/pricing](https://e2b.dev/pricing) |

The estimate covers the agent's sandbox only. A Harbor task with a separate verifier
grades in a second box during `finalize`; that box is not on the agent's trace.
