# Qwen3 30B Thinking math with external Dynamo inference

This four-step scale recipe runs a two-GPU Prime trainer against an external 1-prefill/1-decode Dynamo deployment serving `Qwen/Qwen3-30B-A3B-Thinking-2507`. Prime launches no inference process; Dynamo owns the frontend, sidecars, and vLLM engines.

The same model is used by the existing public examples `qwen30b_math`, `qwen30b_swe`, `multinode/rl.toml`, and `multinode/sft.toml`. Those examples remain the source of truth for larger training and non-Dynamo deployment settings; this recipe adds only the external-Dynamo client shape.

## Prerequisites

Use this version-matched native-gRPC source set:

- Dynamo `feat/dyn-pi-sidecar-v2-review-001` at `836fe81012`
- vLLM `feat/dyn-pi-sidecar-v2-review-001` at `e56ee21b2c`
- Prime `feat/dyn-pi-sidecar-v2-review-001`

Build `vllm-rs` and `dynamo-vllm-sidecar` from those revisions into the same
runtime image. For Kubernetes, start from Dynamo's
`examples/backends/vllm/deploy/sidecar_disagg.yaml` and change the model plus
parallelism/resources for the 30B topology; its adjacent `README.md` documents
the paired-binary image. Then apply the required Prime RL discovery and admin
overlay in [`../README.md`](../README.md). This contract requires both native
gRPC and the Dynamo `/v1/rl/workers` endpoint; a standard Python-only vLLM
worker is not compatible.

Install the math environment:

```bash
prime env install primeintellect/math-env
```

Start a Dynamo 1P/1D deployment and verify its public endpoints:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8001/v1/rl/workers
```

The checked-in localhost URLs are for a colocated dev-pod run. For a DGD, replace them with the frontend services. Replace `weight_broadcast.host` with a trainer address reachable from every sidecar and allow the configured NCCL port through the network policy.

`inference_world_size` must equal the sum of `world_size` in one `/v1/rl/workers` response. This recipe assumes 1P/1D with one rank per engine, for a total of `2`; TP, PP, or managed-DP topologies require the corresponding larger sum.

## Run

```bash
uv run rl @ examples/dynamo/qwen3_30b_Thinking/rl.toml \
  --output-dir outputs/dynamo-qwen3-30b-thinking-math
```

The run is successful when four optimizer steps complete, math rewards are emitted, all worker weight versions advance, and both prefill and decode workers remain healthy.
