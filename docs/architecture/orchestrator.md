# Orchestrator

The orchestrator is a lightweight CPU process that handles the core data and scheduling logic, serving as an intermediary between the trainer and inference service with bidirectional relays. In one direction, it collects rollouts from the inference server, assembles them into packed batches, and dispatches them to the trainer; in the other direction, it relays updated model weights from the trainer to the inference service. The orchestrator utilizes `verifiers` environments to abstract multi-turn rollout generation and scoring. Each training and evaluation environment is exposed as a `vf.EnvServer` as a sidecar to the orchestrator process (default) or as a standalone process (e.g. used in hosted training to run environments in containers).

## Starting the Orchestrator

```bash
uv run orchestrator @ path/to/orch.toml
```

Or as part of the `rl` entrypoint:

```bash
uv run rl \
    --trainer @ path/to/train.toml \
    --orchestrator @ path/to/orch.toml \
    --inference @ path/to/infer.toml
```

See all available configuration options with `uv run orchestrator --help`.
