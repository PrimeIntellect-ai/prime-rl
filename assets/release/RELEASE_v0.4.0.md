# 1. Bring Your Own Algorithms

Researchers can now plug in custom loss functions and advantage functions without modifying the core training code. Define your own RL objectives and advantage estimators, configure them via TOML, and experiment freely.

- **Custom Loss**: provide a per-sequence loss function via `LossInputs` / `LossOutputs` dataclasses
- **Custom Advantage**: provide a per-problem advantage function via `AdvantageInputs` / `AdvantageOutputs` dataclasses
- Configure everything in your TOML config with `type = "custom"`, `import_path` and `kwargs`

```toml
# Custom loss
[loss]
type = "custom"
import_path = "my_module.ppo_clip_loss"
kwargs = { clip_eps = 0.2 }

# Custom advantage
[advantage]
type = "custom"
import_path = "my_module.normalized_advantage"
kwargs = { eps = 1e-8 }
```

See [docs/bring-your-own-algorithms.md](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/bring-your-own-algorithms.md) for full documentation.

[#1715](https://github.com/PrimeIntellect-ai/prime-rl/pull/1715) — Bring your own algorithms

# 2. Multimodal RL Training 

Added experimental support for multimodal reinforcement learning training, enabling RL fine-tuning of vision-language models (VLMs). This opens up new possibilities for training models that can reason over both text and images using reinforcement learning.

Key capabilities:
- Train VLMs with the same GRPO/PPO algorithms used for text-only models
- Multi-turn conversation support for multi-modal interactions, allowing complex dialogue flows with interleaved images and text
- Compatible with existing reward functions and verifiers

[#1680](https://github.com/PrimeIntellect-ai/prime-rl/pull/1680) — Add multimodal training (experimental)
[#1703](https://github.com/PrimeIntellect-ai/prime-rl/pull/1703) — Add multi-turn support for multi-modal RL

# 3. Performance & Parallelism

## Expert Parallelism (EP)

Added support for Expert Parallelism, a distributed training strategy for Mixture of Experts (MoE) models.

[#1595](https://github.com/PrimeIntellect-ai/prime-rl/pull/1595) — Expert Parallelism support
[#1614](https://github.com/PrimeIntellect-ai/prime-rl/pull/1614) — Add CP and EP to benchmarks

## Flash Attention 4

Added FA4 support for fast attention on Blackwell.

[#1726](https://github.com/PrimeIntellect-ai/prime-rl/pull/1726) — Flash Attention 4

## FA3 Ring-Attention Kernel

Previously our ring attention algorithm was still using the Flash Attention 2 kernel. We now allow using FA3 instead for significant speedup on long context training.

[#1727](https://github.com/PrimeIntellect-ai/prime-rl/pull/1727) — Add FA3 ring-attention kernel wrapper and benchmark coverage

## Optimizer State CPU Offload

Offload optimizer states (e.g. Adam first and second moments) to CPU memory. Particularly useful to reduce memory usage when doing RL experiments at smaller scale, allowing large MoE models to fit on a couple of training nodes. The performance reduction is negligible in RL because large batch sizes mean many gradient accumulation steps, and the cost of offloading weights to CPU is amortized.
 
[#1694](https://github.com/PrimeIntellect-ai/prime-rl/pull/1694) — Add optimizer state CPU offload

## 3-Stage Chunked LM Head Loss

Improved memory efficiency for the language model head loss computation via a 3-stage chunked approach. Instead of materializing the full logit tensor, the loss is computed in chunks, reducing peak memory usage. This is especially beneficial for large-vocabulary models where the logit tensor can be a major memory bottleneck during the backward pass.

[#1649](https://github.com/PrimeIntellect-ai/prime-rl/pull/1649) — 3-stage logic for chunked lm head loss

# 4. Other Improvements

- **Elastic Inference Pool**: New elastic inference pool with DNS-based service discovery for dynamic scaling of inference servers at runtime. Add or remove servers without restarting the training loop, with automatic health checking and failover. [#1617](https://github.com/PrimeIntellect-ai/prime-rl/pull/1617), [#1704](https://github.com/PrimeIntellect-ai/prime-rl/pull/1704)
- **Temperature Scheduler**: Control sampling temperature throughout training with various scheduling strategies, enabling curriculum-style exploration. [#1624](https://github.com/PrimeIntellect-ai/prime-rl/pull/1624)
- **JSON Structured Logging**: JSON structured logging for easier log aggregation and analysis in production. [#1681](https://github.com/PrimeIntellect-ai/prime-rl/pull/1681)
- **Gemma3 Support**: Added native support for Gemma3 models. [#1648](https://github.com/PrimeIntellect-ai/prime-rl/pull/1648)
- **Worker Rate Limiting**: Rate limiting for worker job submissions to control dispatch pace. [#1711](https://github.com/PrimeIntellect-ai/prime-rl/pull/1711)
- **K8s Health Probes**: Health probes for inference and trainer, plus parallel pod management for faster scaling. [#1719](https://github.com/PrimeIntellect-ai/prime-rl/pull/1719), [#1718](https://github.com/PrimeIntellect-ai/prime-rl/pull/1718)
- **Multi-run Checkpointing**: Checkpoint support for multiple concurrent training runs. [#1593](https://github.com/PrimeIntellect-ai/prime-rl/pull/1593), [#1632](https://github.com/PrimeIntellect-ai/prime-rl/pull/1632)
- **RunsManager Refactor**: Renamed Runs → RunsManager with hook cleanup, and ability to evict runs with bad batches. [#1619](https://github.com/PrimeIntellect-ai/prime-rl/pull/1619), [#1634](https://github.com/PrimeIntellect-ai/prime-rl/pull/1634)

---

# Breaking Changes

* **vLLM upgraded to 0.14**: Upgraded vLLM dependency to version 0.14. This may require updating your environment. Token chat preprocessing has been aligned with vLLM 0.14 behavior. [#1625](https://github.com/PrimeIntellect-ai/prime-rl/pull/1625), [#1637](https://github.com/PrimeIntellect-ai/prime-rl/pull/1637)

* **Liger kernel model deprecated**: The Liger kernel model implementation has been deprecated. [#1691](https://github.com/PrimeIntellect-ai/prime-rl/pull/1691)

---

# Bug Fixes

[#1717](https://github.com/PrimeIntellect-ai/prime-rl/pull/1717) — Fix race condition
[#1725](https://github.com/PrimeIntellect-ai/prime-rl/pull/1725) — Fix int64 JSON serialization in Chinese character metrics
[#1720](https://github.com/PrimeIntellect-ai/prime-rl/pull/1720) — Handle empty completion_temperatures in prepare_sample
[#1712](https://github.com/PrimeIntellect-ai/prime-rl/pull/1712) — Use stable checkpoints for orchestrator resume
[#1702](https://github.com/PrimeIntellect-ai/prime-rl/pull/1702) — Fix eval watcher only picks up checkpoints in increasing order
[#1693](https://github.com/PrimeIntellect-ai/prime-rl/pull/1693) — Fix NCCL update
[#1690](https://github.com/PrimeIntellect-ai/prime-rl/pull/1690) — Don't create config dir on trainer during config validation
[#1686](https://github.com/PrimeIntellect-ai/prime-rl/pull/1686) — Make NCCL broadcast compatible with DP
[#1683](https://github.com/PrimeIntellect-ai/prime-rl/pull/1683) — Fix bug where hosted RL rollouts were missing final message
[#1670](https://github.com/PrimeIntellect-ai/prime-rl/pull/1670) — Zombie guard on checkpoint
[#1678](https://github.com/PrimeIntellect-ai/prime-rl/pull/1678) — Only master clean weight
[#1665](https://github.com/PrimeIntellect-ai/prime-rl/pull/1665) — Fix support for NCCL mode when resuming from checkpoint
[#1650](https://github.com/PrimeIntellect-ai/prime-rl/pull/1650) — Fix KL mismatch by resetting prefix cache
[#1644](https://github.com/PrimeIntellect-ai/prime-rl/pull/1644) — Fix weight update when enforce_eager=True
[#1642](https://github.com/PrimeIntellect-ai/prime-rl/pull/1642) — Use discovery in eval
[#1636](https://github.com/PrimeIntellect-ai/prime-rl/pull/1636) — Fix CPU offloading
[#1630](https://github.com/PrimeIntellect-ai/prime-rl/pull/1630) — Make search for line more robust
[#1612](https://github.com/PrimeIntellect-ai/prime-rl/pull/1612) — Fix timeout overcounting
[#1609](https://github.com/PrimeIntellect-ai/prime-rl/pull/1609) — Auto-restart env workers on unexpected death
[#1596](https://github.com/PrimeIntellect-ai/prime-rl/pull/1596) — Fix trainer crash when all rollouts in a batch fail
[#1613](https://github.com/PrimeIntellect-ai/prime-rl/pull/1613) — Use step change instead of batch size to demarcate when to update

# Misc

[#1722](https://github.com/PrimeIntellect-ai/prime-rl/pull/1722) — Add AMD Instinct MI300X/MI325X peak FLOPS for MFU calculation
[#1724](https://github.com/PrimeIntellect-ai/prime-rl/pull/1724) — Strip @version suffix from env IDs before loading as Python modules
[#1700](https://github.com/PrimeIntellect-ai/prime-rl/pull/1700) — Track Chinese characters
[#1677](https://github.com/PrimeIntellect-ai/prime-rl/pull/1677) — Wandb async RL inflight
[#1671](https://github.com/PrimeIntellect-ai/prime-rl/pull/1671) — Cancel all rollout eval
[#1640](https://github.com/PrimeIntellect-ai/prime-rl/pull/1640) — Add mismatch-KL stability checks for nightly math runs
[#1635](https://github.com/PrimeIntellect-ai/prime-rl/pull/1635) — Weights reload configuration
[#1638](https://github.com/PrimeIntellect-ai/prime-rl/pull/1638) — Add INFO log when orchestrator resumes after checkpoint wait
[#1631](https://github.com/PrimeIntellect-ai/prime-rl/pull/1631) — Ensure eval results upload before existing subprocess
[#1629](https://github.com/PrimeIntellect-ai/prime-rl/pull/1629) — Assert when only trainer or orchestrator wandb is configured
[#1622](https://github.com/PrimeIntellect-ai/prime-rl/pull/1622) — Add retry with exponential backoff for empty training batches
[#1601](https://github.com/PrimeIntellect-ai/prime-rl/pull/1601) — Add health endpoint for worker nodes in multi-node training
[#1604](https://github.com/PrimeIntellect-ai/prime-rl/pull/1604) — Check for current step based on progress to know what is valid for this step
[#1543](https://github.com/PrimeIntellect-ai/prime-rl/pull/1543) — Add option to skip has model check
[#1608](https://github.com/PrimeIntellect-ai/prime-rl/pull/1608) — Improve log message on orchestrator for hosted RL
[#1692](https://github.com/PrimeIntellect-ai/prime-rl/pull/1692) — Remove CC check for grouped mm
[#1699](https://github.com/PrimeIntellect-ai/prime-rl/pull/1699) — Add HF Hub timeout defaults to Dockerfile
[#1653](https://github.com/PrimeIntellect-ai/prime-rl/pull/1653) — Add missing [ckpt] section to reverse_text rl.toml
[#1597](https://github.com/PrimeIntellect-ai/prime-rl/pull/1597) — Add k8s doc to docs folders and update mint config
[#1627](https://github.com/PrimeIntellect-ai/prime-rl/pull/1627) — Add AGENTS.md and CLAUDE.md
[#1633](https://github.com/PrimeIntellect-ai/prime-rl/pull/1633) — Pin Ruff in pre-commit + add Ruff format to CI

# Contributors

@Jackmin801, @samsja, @JannikSt, @hallerite, @S1ro1, @manveerxyz, @kalomaze, @windlgrass, @rasdani, @nph4rd, @minpeter, @mikasenghaas, @faresobeid, @eexwhyzee, @dzautner, @DamianB-BitFlipper, @d42me


