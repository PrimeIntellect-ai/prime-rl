# RL (Single Node)

## Single GPU

If you only have access to a single GPU, you may still be able to run small RL experiments. Configure your inference server to use only a fraction of the available memory to leave some space for the trainer.

```bash
bash scripts/tmux.sh
```

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 0 \
  --inference.gpu-memory-utilization 0.5
```

Make sure to tune the `--gpu-memory-utilization` value such that you have enough GPU memory for the RL trainer.

You can also set this up by starting each submodule manually:

```bash
# Inference pane
uv run inference @ path/to/infer.toml --gpu-memory-utilization 0.5

# Orchestrator pane
uv run orchestrator @ path/to/orch.toml

# Trainer pane
uv run trainer @ path/to/train.toml
```

## Multi-GPU

For single-node training, we recommend using the `rl` entrypoint to conveniently start all components.

By default, the inference server starts on GPU ID 0 and the trainer on GPU ID 1.

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml
```

You can configure the GPU IDs to use for the inference server and the trainer. For example, to run the inference server on GPUs 0-5 with data parallelism and the trainer on GPUs 6-7:

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --inference-gpu-ids 0,1,2,3,4,5 \
  --trainer-gpu-ids 6,7 \
  --inference.parallel.dp 6
```

## Parallel Experiments

For quick ablations, it can be more efficient to parallelize experiments within a node (e.g. split your GPUs to run two experiments in parallel).

Start the first experiment in a tmux session `exp1` (uses GPUs 0-1):

```bash
bash scripts/tmux.sh -s exp1 -o outputs1
```

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --output-dir outputs1
```

Start a second experiment in `exp2` (uses GPUs 2-3), with a different server port:

```bash
bash scripts/tmux.sh -s exp-2 -o outputs2
```

```bash
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --inference @ path/to/infer.toml \
  --inference-gpu-ids 2 \
  --trainer-gpu-ids 3 \
  --inference.server.port 8001 \
  --orchestrator.client.base-url http://localhost:8001/v1 \
  --output-dir outputs2
```
