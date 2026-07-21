# Qwen3 0.6B math with external Dynamo inference

This four-step smoke recipe runs the Prime trainer and orchestrator locally while generation is served by an already-running Dynamo frontend, vLLM sidecar, and vLLM engine. Prime does not launch local inference, so the configuration intentionally has no `[inference]` block.

## Prerequisites

Use this version-matched native-gRPC source set:

- Dynamo `feat/dyn-pi-sidecar-v2-review-001` at `836fe81012`
- vLLM `feat/dyn-pi-sidecar-v2-review-001` at `e56ee21b2c`
- Prime `feat/dyn-pi-sidecar-v2-review-001`

Build `vllm-rs` and `dynamo-vllm-sidecar` from those revisions into the same
runtime image. For Kubernetes, start from Dynamo's
`examples/backends/vllm/deploy/sidecar_agg.yaml`; its adjacent `README.md`
documents the paired-binary image. Then apply the required Prime RL discovery
and admin overlay in [`../README.md`](../README.md). This contract requires both
native gRPC and the Dynamo `/v1/rl/workers` endpoint; a standard Python-only
vLLM worker is not compatible.

Install the math environment:

```bash
prime env install primeintellect/math-env
```

Start an aggregated DP1 Dynamo deployment for `Qwen/Qwen3-0.6B`. The orchestrator waits for both model publication and worker discovery. These requests are useful diagnostics:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8001/v1/rl/workers
```

The checked-in URLs assume Dynamo is reachable from the trainer through localhost, as in a shared dev pod. For a remote DGD, replace both URLs with its frontend services. Also replace `weight_broadcast.host` with a trainer hostname or IP reachable from every sidecar; localhost is not valid across pods or nodes.

`inference_world_size` must equal the sum of `world_size` in one `/v1/rl/workers` response. This recipe assumes one aggregated DP1 engine and therefore uses `1`.

## Run

```bash
uv run rl @ examples/dynamo/qwen3_06b_math/rl.toml \
  --output-dir outputs/dynamo-qwen3-06b-math
```

The run is successful when four optimizer steps complete, the verifier reports math rewards, weight versions advance after each update, and the Dynamo workers remain healthy.
