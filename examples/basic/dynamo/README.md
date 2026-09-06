# Dynamo RL

This example runs five steps of GRPO training on the GSM8K math taskset with `Qwen/Qwen3-0.6B`, one inference GPU, one trainer GPU, and NCCL weight transfer.

Prime-RL owns the full local process tree. The existing `rl` launcher starts the math environment server, orchestrator, trainer, and one `inference` service. With `backend = "dynamo"`, that inference service supervises:

- `python -m dynamo.frontend` on port 8000.
- `python -m dynamo.vllm --enable-rl` on the inference GPU.

The integrated Dynamo vLLM worker exposes the discovery and administration routes, so this example does not require a separately built `dynamo-vllm-sidecar`.
Managed Dynamo currently supports NCCL and NIXL weight transfer with one inference rank. Filesystem transfer, LoRA updates, sampling-mask capture, routed-expert capture, KV-cache offload, and multi-node workers are outside this first increment and fail with explicit configuration errors.

The sampling override sets `top_k = 0`, Dynamo's unsigned representation for disabled top-k sampling. Verifiers otherwise sends the equivalent vLLM sentinel `-1`, which the Dynamo frontend cannot deserialize as an unsigned integer.

## Install

Initialize the repository and install the GPU, Dynamo, and environment dependencies:

```bash
git submodule update --init --recursive
uv sync --all-extras --all-packages
```

## Run locally

From the repository root, make two GPUs visible and run the example:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --locked --all-extras --all-packages rl @ examples/basic/dynamo/rl.toml
```

The launcher assigns visible GPU 0 to Dynamo inference and visible GPU 1 to the trainer. It starts every required process, waits for the Dynamo frontend and worker through the orchestrator's normal readiness path, runs exactly five optimizer steps, and cleans up the managed processes when training finishes or a child fails.

The resolved configuration and logs are written under `outputs/<run-name>/`. Dynamo frontend and worker output is captured in the launcher's `inference.log`.

## Endpoint contract

The example uses:

- OpenAI-compatible inference at `http://127.0.0.1:8000/v1`.
- Dynamo worker discovery at `http://127.0.0.1:8001/v1/rl/workers`.
- The worker system server at `http://127.0.0.1:8081`.

Prime-RL derives the discovery URL from the model client URL by incrementing the explicit port and defaults the worker system port to 81 ports above the frontend, producing 8001 and 8081 for this example. Override `orchestrator.model.client.dynamo.discovery_url` when the discovery ports are not adjacent. Override the worker system port with `inference.env_vars.DYN_SYSTEM_PORT`; all three ports must be distinct.

## Run with Slurm

The same managed process topology works with Prime-RL's existing single-node Slurm launcher. Apply the included overlay after the base configuration:

```bash
uv run --locked --all-extras --all-packages rl @ examples/basic/dynamo/rl.toml @ examples/basic/dynamo/slurm.toml
```

Set the partition, account, or project directory in `slurm.toml` for the target cluster. The example requests two GPUs on one node. Multi-node Dynamo workers are intentionally outside this first managed-worker increment.
