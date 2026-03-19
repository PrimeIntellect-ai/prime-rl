# Quickstart

This guide walks you through running your first SFT and RL training.

## SFT

On a single GPU:

```bash
uv run sft @ examples/reverse_text/configs/sft.toml
```

With multiple GPUs:

```bash
uv run torchrun --nproc-per-node 8 src/prime_rl/trainer/sft/train.py @ examples/reverse_text/configs/sft.toml
```

## RL

RL training requires three components: an inference server, an orchestrator, and a trainer. The `rl` entrypoint starts all three on a single node.

```bash
uv run rl \
    --trainer @ examples/reverse_text/configs/train.toml \
    --orchestrator @ examples/reverse_text/configs/orch.toml \
    --inference @ examples/reverse_text/configs/infer.toml
```

We recommend using the tmux helper to monitor all components:

```bash
bash scripts/tmux.sh
```

This creates a tmux session with three panes (trainer, orchestrator, inference) that stream their respective logs.

## What's Next

- [Config System](config-system.md) — learn how to configure runs
- [Development](development.md) — debug with fake data, test at small scale
- [Architecture](../architecture/overview.md) — understand how the components work together
- [Single Node](../single-node/overview.md) — explore everything you can do on one machine
- [Multi Node](../multi-node/overview.md) — scale to multiple nodes
