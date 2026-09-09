# Dynamo RL

This example runs five steps of GRPO training on the Hendrycks math environment with `Qwen/Qwen3-0.6B`, one trainer GPU, one external Dynamo inference worker, and NCCL weight transfer.

Prime-RL does not launch Dynamo from this configuration. Run the frontend, worker, and trainer as separate processes. This example requires two GPUs: GPU 0 for Prime-RL training and GPU 1 for Dynamo inference.

## Start Dynamo

Install Dynamo with its vLLM backend and activate its environment. Then start the RL-enabled frontend in the first terminal:

```bash
DYN_HTTP_HOST=127.0.0.1 DYN_ENABLE_RL=true DYN_RL_PORT=8001 python -m dynamo.frontend
```

Start the RL-enabled vLLM worker on GPU 1 in a second terminal:

```bash
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_HOST=127.0.0.1 DYN_SYSTEM_PORT=8081 python -m dynamo.vllm \
  --model Qwen/Qwen3-0.6B \
  --enable-rl
```

The `--enable-rl` flag enables worker discovery and the administration routes used for NCCL weight updates. Keep GPU 0 out of `CUDA_VISIBLE_DEVICES` for this process so it remains available to the Prime-RL trainer.

These commands bind the HTTP, discovery, and worker administration endpoints to loopback. If Prime-RL and Dynamo run on different hosts, expose these endpoints only on a trusted control network; configured client headers are also sent to the discovery endpoint.

## Endpoint contract

The default configuration expects:

- The Dynamo OpenAI-compatible frontend at `http://127.0.0.1:8000/v1`.
- The Dynamo RL discovery endpoint at `http://127.0.0.1:8001/v1/rl/workers`.
- Exactly one discovered worker for `Qwen/Qwen3-0.6B` with an admin URL and `world_size = 1`.
- A vLLM worker with the Prime-RL NCCL weight-update extension enabled.

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
uv run rl @ examples/basic/dynamo/rl.toml
```

The example stops after five optimizer steps. Outputs are written under `outputs/` unless `output_dir` is overridden.
