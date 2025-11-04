# Docs

This directory maintains the documentation for PRIME-RL. It is organized into the following sections:

- [**Entrypoints**](docs/entrypoints.md) - Overview of the main components (orchestrator, trainer, inference) and how to run SFT, RL, and evals
- [**Configs**](docs/configs.md) - Configuration system using TOML files, CLI arguments, and environment variables
- [**Environments**](docs/environments.md) - Installing and using verifiers environments from the Environments Hub
- [**Async Training**](docs/async.md) - Understanding asynchronous off-policy training and step semantics
- [**Logging**](docs/logging.md) - Logging with loguru, torchrun, and Weights & Biases
- [**Checkpointing**](docs/checkpointing.md) - Saving and resuming training from checkpoints
- [**Benchmarking**](docs/benchmarking.md) - Performance benchmarking and throughput measurement
- [**Deployment**](docs/deployment.md) - Training deployment on single-GPU, multi-GPU, and multi-node clusters
- [**Troubleshooting**](docs/troubleshooting.md) - Common issues and their solutions