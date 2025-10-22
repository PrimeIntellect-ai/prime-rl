# Debugging

this page reference to the debug configs in the `configs/debug` and `configs/debug_moe` directories. 

This config are designed for debugging and reducing the iteration loop for development. Most of this configuration are also tested in our CI pipeline.


All dense debug are using 

## SFT
dense model


```bash
uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/prime_rl/trainer/sft/train.py @ configs/debug/sft/train.toml
```

moe model

```bash
uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/prime_rl/trainer/sft/train.py @ configs/debug_moe/sft/train.toml
```

## RL training only

dense model 

```bash
uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/prime_rl/trainer/rl/train.py @ configs/debug/rl/train.toml
```

moe model

```bash
uv run torchrun --local-ranks-filter 0 --nproc_per_node=2 src/prime_rl/trainer/rl/train.py @ configs/debug_moe/rl/train.toml
```

## Inference

```bash
uv run inference @ configs/debug/infer.toml
```

there is no inference server for moe model yet.

## Orchestrator

```bash
uv run orchestrator @ configs/debug/orch.toml
```