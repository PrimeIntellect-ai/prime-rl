# Docs

This directory maintains the documentation for PRIME-RL. It is organized into the following sections:

- [**Entrypoints**](entrypoints.md) - Overview of the main components (orchestrator, trainer, inference) and how to run SFT, RL, and evals
- [**Configs**](configs.md) - Configuration system using TOML files, CLI arguments, and environment variables
- [**Environments**](environments.md) - Installing and using verifiers environments from the Environments Hub
- [**Async Training**](async.md) - Understanding asynchronous off-policy training and step semantics
- [**Logging**](logging.md) - Logging with loguru, torchrun, and Weights & Biases
- [**Runs**](runs.md) - Multi-run training with the Runs object for concurrent LoRA adapters
- [**Checkpointing**](checkpointing.md) - Saving and resuming training from checkpoints
- [**Benchmarking**](benchmarking.md) - Performance benchmarking and throughput measurement
- [**Deployment**](deployment.md) - Training deployment on single-GPU, multi-GPU, and multi-node clusters
- [**Kubernetes**](kubernetes.md) - Deploying PRIME-RL on Kubernetes with Helm
- [**Troubleshooting**](troubleshooting.md) - Common issues and their solutions