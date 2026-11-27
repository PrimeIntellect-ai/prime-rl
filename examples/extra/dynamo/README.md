# Dynamo RL

This example runs five steps of GRPO training on the Hendrycks math environment with `Qwen/Qwen3-0.6B`, one trainer GPU, one external Dynamo inference worker, and [ModelExpress](../../../docs/model-express.md) weight transfer.

Prime-RL does not launch Dynamo from this configuration. Run the frontend, worker, and trainer as separate processes. This example requires two GPUs: GPU 0 for Prime-RL training and GPU 1 for Dynamo inference.

## Start Dynamo

Dynamo's frontend and vLLM sidecar come with the `dynamo` extra, and vLLM's Rust frontend, `vllm-rs`, ships in the vLLM wheel. `uv sync --all-extras` installs both. Dynamo needs three processes on the inference host: the frontend, a vLLM engine served by `vllm-rs`, and a sidecar that registers the engine with Dynamo. File-based discovery keeps them on one host without etcd or NATS. Export the same settings in every terminal:

```bash
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/dynamo-discovery
```

Start the frontend with RL worker discovery and the token-in, token-out route Prime-RL samples from:

```bash
DYN_HTTP_HOST=127.0.0.1 DYN_HTTP_PORT=8000 DYN_ENABLE_RL=true DYN_RL_PORT=8001 \
DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE=1 uv run python -m dynamo.frontend
```

Start the engine on GPU 1 with the Prime-RL ModelExpress weight-update extension. `VLLM_SERVER_DEV_MODE=1` enables the `/pause`, `/resume` and `/collective_rpc` administration routes on its HTTP port:

```bash
VLLM_RS=$(uv run python -c "import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), 'vllm-rs'))")
CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 uv run "$VLLM_RS" serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8100 --grpc-port 50051 --max-model-len 4096 --reasoning-parser none \
  -- --worker-extension-cls prime_rl.inference.vllm.worker.modelexpress.ModelExpressWeightUpdateWorker
```

Arguments after `--` go to the vLLM engine. Keep GPU 0 out of `CUDA_VISIBLE_DEVICES` so it remains available to the Prime-RL trainer.

Start the sidecar. It serves the engine through the frontend and advertises the engine's HTTP origin as the worker's administration URL:

```bash
DYN_SYSTEM_HOST=127.0.0.1 DYN_SYSTEM_PORT=8081 uv run python -m dynamo.vllm.sidecar --enable-rl \
  --grpc-endpoint 127.0.0.1:50051 --vllm-http-endpoint http://127.0.0.1:8100 --vllm-rl-world-size 1
```

These commands bind the frontend, discovery, and worker administration endpoints to loopback. If Prime-RL and Dynamo run on different hosts, expose these endpoints only on a trusted control network; configured client headers are also sent to the discovery endpoint.

## Endpoint contract

The default configuration expects:

- The Dynamo frontend at `http://127.0.0.1:8000/v1`, serving `/inference/v1/generate`.
- The Dynamo RL discovery endpoint at `http://127.0.0.1:8001/v1/rl/workers`, listing each worker for `Qwen/Qwen3-0.6B` with its `admin_base_url` and `world_size`.
- Every worker running the Prime-RL ModelExpress weight-update extension.

Verify both endpoints before starting training:

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8001/v1/rl/workers
```

## Discovery URL

The example enables Dynamo with:

```toml
[orchestrator.model.client]
base_url = "http://127.0.0.1:8000/v1"

[orchestrator.model.client.dynamo]
enabled = true
```

When `discovery_url` is omitted, Prime-RL removes the path from `base_url` and increments its explicit non-default port by one. Here, `http://127.0.0.1:8000/v1` becomes `http://127.0.0.1:8001`.

If your Dynamo frontend and discovery service do not use adjacent ports, configure the discovery endpoint explicitly:

```toml
[orchestrator.model.client.dynamo]
enabled = true
discovery_url = "http://dynamo.example:9000"
```

## Run training

After both endpoint checks pass, run Prime-RL from the repository in a third terminal. With no managed `[inference]` section, Prime-RL assigns GPU 0 to its single trainer process and leaves the external Dynamo worker on GPU 1:

```bash
uv run rl @ examples/extra/dynamo/rl.toml
```

The example stops after five optimizer steps. Outputs are written under `outputs/` unless `output_dir` is overridden.

## Sampling

Dynamo's sidecar rejects Prime-RL's request extensions (`extra_args`), so train sources that sample from the policy must use plain sampling: no truncation (`top_p` below 1 or `top_k`, which need sampling replay) and no logprob head (`sampling.logprobs`, which score centering sets). Config validation rejects these with Dynamo. Tokens carry no weight versions, so staleness counts each rollout as old as its dispatch version.

## Weight transfer

The example sends weights through [ModelExpress](../../../docs/model-express.md) on port 8011, since the default ModelExpress port, 8001, is the Dynamo discovery port here:

```toml
[weight_broadcast]
type = "modelexpress"
port = 8011
```

`uv run rl` starts the ModelExpress server. Prime-RL accepts any number of discovered workers and any worker `world_size`, and rediscovers them before every weight update: a worker that joins is initialized and receives the next update, and a worker that leaves is dropped. A worker that joins between updates serves its launch weights until then.
