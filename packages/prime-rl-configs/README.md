# prime-rl-configs

Slim config schema for [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl), with no GPU or ML deps.

`pip install prime-rl-configs` gives you `prime_rl.configs.*` (RL/SFT/inference/orchestrator/trainer/env-server schemas) without pulling in `torch`, `vllm`, `transformers`, `wandb`, etc. The full training stack lives in `prime-rl`, which depends on this package.
