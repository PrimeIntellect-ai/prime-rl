# 1. Disaggregated Prefill-Decode Inference

Added support for disaggregated prefill-decode inference with multi-replica support. This architecture separates prefill and decode phases across dedicated GPU pools, improving throughput and latency for large-scale RL training. Includes vLLM router integration for intelligent request routing across replicas.

- Separate prefill and decode workers for optimal GPU utilization
- Multi-replica support for horizontal scaling
- Router URL support for elastic inference pool integration

[#2030](https://github.com/PrimeIntellect-ai/prime-rl/pull/2030) — Disaggregated prefill-decode inference with multi-replica support
[#2049](https://github.com/PrimeIntellect-ai/prime-rl/pull/2049) — Add router_url support for elastic inference pool

# 2. New Model Support

## GLM5

Added support for GLM5 models, including deep-gemm dependency for FP8 operations and AFMoE flash attention.

[#1827](https://github.com/PrimeIntellect-ai/prime-rl/pull/1827) — GLM5 support
[#1884](https://github.com/PrimeIntellect-ai/prime-rl/pull/1884) — Add deep_gemm as dependency for GLM5
[#1766](https://github.com/PrimeIntellect-ai/prime-rl/pull/1766) — Add flash attention for AFMoE

## Qwen3.5 MoE

Added Qwen3.5 MoE model support including EP, VLM weight broadcast, hybrid context parallelism (DeltaNet + attention), and LoRA compatibility.

[#1946](https://github.com/PrimeIntellect-ai/prime-rl/pull/1946) — Qwen3.5 support
[#2026](https://github.com/PrimeIntellect-ai/prime-rl/pull/2026) — Qwen3.5 MoE model support with EP and VLM weight broadcast
[#2080](https://github.com/PrimeIntellect-ai/prime-rl/pull/2080) — Hybrid CP for Qwen3.5 MoE (DeltaNet + attention)
[#2019](https://github.com/PrimeIntellect-ai/prime-rl/pull/2019) — Monkeypatch for Qwen3.5 LoRA
[#2027](https://github.com/PrimeIntellect-ai/prime-rl/pull/2027) — Patch sharded LoRA slice_lora_a for Qwen3.5 MoE

## MiniMax M2.5

Added MiniMax M2.1 MoE model support with LoRA compatibility fixes for vLLM inference.

[#1773](https://github.com/PrimeIntellect-ai/prime-rl/pull/1773) — MiniMax M2.1 MoE model support
[#1831](https://github.com/PrimeIntellect-ai/prime-rl/pull/1831) — Fix MiniMax M2 LoRA compatibility in vLLM inference

## Nemotron-H

Added NemotronH (Nemotron-3-Super-120B-A12B) support, a hybrid Mamba-Transformer model, including tool call parser integration and dimension mismatch fixes for non-120B variants.

[#2046](https://github.com/PrimeIntellect-ai/prime-rl/pull/2046) — NemotronH model support
[#2089](https://github.com/PrimeIntellect-ai/prime-rl/pull/2089) — Add NemotronH to tool call parser map
[#2110](https://github.com/PrimeIntellect-ai/prime-rl/pull/2110) — Fix Nemotron-H Mamba layer dimension mismatch for non-120B models

## GPT-OSS

Added GPT-OSS model support with LoRA and DP fixes.

[#2073](https://github.com/PrimeIntellect-ai/prime-rl/pull/2073) — GPT-OSS support with LoRA + DP fixes

# 3. Multi Env Worker

New multi-environment worker architecture that isolates environment execution from scheduling logic. Environment servers now run as separate processes, improving fault isolation and enabling independent scaling of env execution.

[#1714](https://github.com/PrimeIntellect-ai/prime-rl/pull/1714) — Env worker integration
[#2083](https://github.com/PrimeIntellect-ai/prime-rl/pull/2083) — Multi env worker
[#1816](https://github.com/PrimeIntellect-ai/prime-rl/pull/1816) — Env server recovery
[#1939](https://github.com/PrimeIntellect-ai/prime-rl/pull/1939) — Fix env server task cancellation
[#2028](https://github.com/PrimeIntellect-ai/prime-rl/pull/2028) — Cleanup env subprocesses on orchestrator crash

# 4. SFT Improvements

## SFT LoRA

Added LoRA support for supervised fine-tuning, enabling parameter-efficient training.

[#1849](https://github.com/PrimeIntellect-ai/prime-rl/pull/1849) — SFT LoRA support

## SFT Distillation

Bug fixes, VLM support, and pretokenization optimization for SFT distillation.

[#2053](https://github.com/PrimeIntellect-ai/prime-rl/pull/2053) — SFT distillation: bug fixes, VLM support, and pretokenization optimization

## Fused Linear Cross-Entropy Loss

Memory-efficient fused linear cross-entropy loss for SFT, reducing peak memory usage.

[#1958](https://github.com/PrimeIntellect-ai/prime-rl/pull/1958) — Fused linear cross-entropy loss for SFT

## SFT Validation

Added validation eval with `val_data` support for monitoring training quality.

[#1850](https://github.com/PrimeIntellect-ai/prime-rl/pull/1850) — SFT validation eval with val_data

# 5. Performance

## Selective Activation Checkpointing

Selectively checkpoint activations per layer, reducing memory while preserving compute for layers that don't need recomputation.

[#2055](https://github.com/PrimeIntellect-ai/prime-rl/pull/2055) — Selective activation checkpointing

## FP8 Weight Transfer

Integrated FP8 weight transfer format for faster model weight synchronization between trainer and inference.

[#2038](https://github.com/PrimeIntellect-ai/prime-rl/pull/2038) — Integrate FP8 weight transfer format

## Sequence-Chunked Fused LM Head

Switched fused LM head from token chunking to sequence chunking for better memory efficiency.

[#1987](https://github.com/PrimeIntellect-ai/prime-rl/pull/1987) — Switch fused LM head to sequence chunking

## AFMoE Ring Attention

Added ring attention support for AFMoE (Alternating Full MoE) architectures.

[#1848](https://github.com/PrimeIntellect-ai/prime-rl/pull/1848) — AFMoE ring attention support

## VLM Performance

Multiple optimizations for vision-language model training: parallel image preprocessing, image deduplication, disk-backed offloading, and async preprocessing.

[#1951](https://github.com/PrimeIntellect-ai/prime-rl/pull/1951) — Parallelize VLM image preprocessing across threads
[#1935](https://github.com/PrimeIntellect-ai/prime-rl/pull/1935) — Deduplicate VLM image bytes in orchestrator cache
[#1923](https://github.com/PrimeIntellect-ai/prime-rl/pull/1923) — Serialize VLM pixel_values as raw bytes for 8-10x faster preprocessing
[#2006](https://github.com/PrimeIntellect-ai/prime-rl/pull/2006) — Disk-backed image offloading for VLM memory
[#2065](https://github.com/PrimeIntellect-ai/prime-rl/pull/2065) — Run VLM image preprocessing in thread to unblock event loop

## Quack Integration

Integrated Quack kernels for RMS norm and SFT loss.

[#2102](https://github.com/PrimeIntellect-ai/prime-rl/pull/2102) — Use Quack for RMS norm and SFT loss

# 6. Infrastructure & Deployment

## SLURM Entrypoint

Unified SLURM entrypoint for both RL and SFT with Jinja2 template-based sbatch generation. Supports single-node deployments and configurable pre-run commands.

[#1774](https://github.com/PrimeIntellect-ai/prime-rl/pull/1774) — SLURM entrypoint with Jinja2 template
[#1832](https://github.com/PrimeIntellect-ai/prime-rl/pull/1832) — SFT SLURM entrypoint
[#1859](https://github.com/PrimeIntellect-ai/prime-rl/pull/1859) — Unify RL and SFT SLURM entrypoints
[#1988](https://github.com/PrimeIntellect-ai/prime-rl/pull/1988) — Add pre_run_command and common SLURM scheduling options

## Arm64 Support

Added aarch64 (arm64) support to Docker builds and dependencies.

[#1933](https://github.com/PrimeIntellect-ai/prime-rl/pull/1933) — Arm64 support for Dockerfile and deps

## Config Migration to pydantic-config

Migrated the configuration system to pydantic-config with JSON dict CLI support, cleaner discriminated unions, and consolidated config modules.

[#1915](https://github.com/PrimeIntellect-ai/prime-rl/pull/1915) — Migrate config system to pydantic_config
[#1871](https://github.com/PrimeIntellect-ai/prime-rl/pull/1871) — Consolidate config modules into prime_rl.configs
[#1878](https://github.com/PrimeIntellect-ai/prime-rl/pull/1878) — Replace callable discriminator with field values

## Platform Integration

Added platform integration for centralized management.

[#1896](https://github.com/PrimeIntellect-ai/prime-rl/pull/1896) — Platform integration

## Docs Revamp

Revamped documentation and README.

[#2116](https://github.com/PrimeIntellect-ai/prime-rl/pull/2116) — Revamp docs and readme

# 7. Other Improvements

- **Individual Rollouts**: Schedule and track rollouts individually for finer-grained control. [#1865](https://github.com/PrimeIntellect-ai/prime-rl/pull/1865)
- **TITO Default**: Text-In-Text-Out (TITO) is now the default inference endpoint. [#1851](https://github.com/PrimeIntellect-ai/prime-rl/pull/1851)
- **Group Relative Reward Rescaling**: Added GRRR as a length penalty option. [#2029](https://github.com/PrimeIntellect-ai/prime-rl/pull/2029)
- **DP-Rank Routing**: Route rollouts based on data-parallel rank. [#1940](https://github.com/PrimeIntellect-ai/prime-rl/pull/1940)
- **Router Replay**: Replay routing decisions for debugging and reproducibility. [#1807](https://github.com/PrimeIntellect-ai/prime-rl/pull/1807)
- **Gibberish & Repetition Filtering**: Filter out low-quality rollouts. [#1746](https://github.com/PrimeIntellect-ai/prime-rl/pull/1746)
- **Weights-Only Checkpointing**: Save just model weights without optimizer state. [#2033](https://github.com/PrimeIntellect-ai/prime-rl/pull/2033)
- **Per-Env Metrics**: Logging and metrics broken down per training environment. [#1989](https://github.com/PrimeIntellect-ai/prime-rl/pull/1989), [#2070](https://github.com/PrimeIntellect-ai/prime-rl/pull/2070)
- **Eval Metrics**: Log `eval/{env}/failed_rollouts` to W&B and add `eval/samples` table. [#2123](https://github.com/PrimeIntellect-ai/prime-rl/pull/2123), [#2124](https://github.com/PrimeIntellect-ai/prime-rl/pull/2124)
- **EP Inference Support**: Expert parallelism at inference time. [#1860](https://github.com/PrimeIntellect-ai/prime-rl/pull/1860)
- **Multi-Node EP**: Support for expert parallelism across multiple nodes. [#1894](https://github.com/PrimeIntellect-ai/prime-rl/pull/1894)
- **IPO Default Algorithm**: Changed default RL algorithm to IPO. [#1930](https://github.com/PrimeIntellect-ai/prime-rl/pull/1930)
- **Inference Entrypoint**: Standalone entrypoint for inference server. [#1898](https://github.com/PrimeIntellect-ai/prime-rl/pull/1898)
- **Auto-Detect Tool Call Parser**: Automatically detect the correct tool call parser for a model. [#1844](https://github.com/PrimeIntellect-ai/prime-rl/pull/1844)
- **Configurable Max Retries Per Env**: Set retry limits per training environment. [#2025](https://github.com/PrimeIntellect-ai/prime-rl/pull/2025)
- **Sample Ratio**: Gradual rollout data control via `sample_ratio`. [#2023](https://github.com/PrimeIntellect-ai/prime-rl/pull/2023)
- **Shared Wandb Mode**: Share a single Wandb run across components. [#2044](https://github.com/PrimeIntellect-ai/prime-rl/pull/2044)
- **Freeze MoE Router**: Option to freeze MoE router weights during training. [#1836](https://github.com/PrimeIntellect-ai/prime-rl/pull/1836)
- **Token Batch Size**: Option for token-level batch sizing. [#1855](https://github.com/PrimeIntellect-ai/prime-rl/pull/1855)
- **Prefill vs Decode Token Tracking**: Track prefill and decode tokens separately. [#1797](https://github.com/PrimeIntellect-ai/prime-rl/pull/1797)
- **Accept `messages` Column for SFT**: Support `messages` column format in SFT datasets. [#2074](https://github.com/PrimeIntellect-ai/prime-rl/pull/2074)
- **Dump Config Flag**: `--dump-config` flag for RL command. [#1771](https://github.com/PrimeIntellect-ai/prime-rl/pull/1771)
- **Separate Checkpoint/Weight Paths**: Save checkpoints and weights to separate paths. [#2018](https://github.com/PrimeIntellect-ai/prime-rl/pull/2018)

---

# Breaking Changes

* **vLLM upgraded to 0.17**: Upgraded vLLM from 0.14 to 0.16 to 0.17 stable. This may require updating your environment. [#1731](https://github.com/PrimeIntellect-ai/prime-rl/pull/1731), [#1980](https://github.com/PrimeIntellect-ai/prime-rl/pull/1980)
* **Transformers upgraded to v5**: Bumped transformers to version 5. [#1731](https://github.com/PrimeIntellect-ai/prime-rl/pull/1731)
* **Config system migrated to pydantic-config**: The TOML config system now uses pydantic-config. Existing configs may need minor adjustments. [#1915](https://github.com/PrimeIntellect-ai/prime-rl/pull/1915)
* **Default algorithm changed to IPO**: The default RL algorithm is now IPO instead of GRPO. [#1930](https://github.com/PrimeIntellect-ai/prime-rl/pull/1930)
* **TITO is now the default endpoint**: Text-In-Text-Out is the default inference mode. [#1851](https://github.com/PrimeIntellect-ai/prime-rl/pull/1851)
* **Loss normalization changed**: Loss now uses equal weight per unmasked token and token-level normalization for all loss types. [#1961](https://github.com/PrimeIntellect-ai/prime-rl/pull/1961), [#2034](https://github.com/PrimeIntellect-ai/prime-rl/pull/2034)
* **Removed TP trainer option**: Tensor parallelism for the trainer has been removed. [#2109](https://github.com/PrimeIntellect-ai/prime-rl/pull/2109)
* **Removed model map**: Model map has been replaced by auto-detection. [#1842](https://github.com/PrimeIntellect-ai/prime-rl/pull/1842)

---

# Bug Fixes

[#2129](https://github.com/PrimeIntellect-ai/prime-rl/pull/2129) — Remove flaky grad norm check from test_nemotron_h
[#2112](https://github.com/PrimeIntellect-ai/prime-rl/pull/2112) — Add weights_only=False to torch.load for checkpoint resume
[#2100](https://github.com/PrimeIntellect-ai/prime-rl/pull/2100) — Fix Mito RL training
[#2097](https://github.com/PrimeIntellect-ai/prime-rl/pull/2097) — Fix hybrid CP for Qwen3.5 MoE (DeltaNet + attention)
[#2092](https://github.com/PrimeIntellect-ai/prime-rl/pull/2092) — Terminate process on fatal orchestrator error
[#2091](https://github.com/PrimeIntellect-ai/prime-rl/pull/2091) — Fix mamba-ssm build by using CUDA torch in build env
[#2086](https://github.com/PrimeIntellect-ai/prime-rl/pull/2086) — Handle eval rollout failures without crashing the orchestrator
[#2079](https://github.com/PrimeIntellect-ai/prime-rl/pull/2079) — Don't constrain VLM dtype for SFT
[#2056](https://github.com/PrimeIntellect-ai/prime-rl/pull/2056) — Use mean().mean() for console metrics to match wandb
[#2035](https://github.com/PrimeIntellect-ai/prime-rl/pull/2035) — Handle DTensor weights in MultiLoRAGroupedExperts for EP
[#2016](https://github.com/PrimeIntellect-ai/prime-rl/pull/2016) — Re-schedule errored rollouts
[#2014](https://github.com/PrimeIntellect-ai/prime-rl/pull/2014) — Re-schedule empty trajectories instead of filtering after group completion
[#2004](https://github.com/PrimeIntellect-ai/prime-rl/pull/2004) — Avoid duplicate policy updates during scheduler cancellation
[#2003](https://github.com/PrimeIntellect-ai/prime-rl/pull/2003) — Filter empty trajectory rollouts in scheduler
[#1999](https://github.com/PrimeIntellect-ai/prime-rl/pull/1999) — Patch HF tokenizer 'Already borrowed' crash under concurrent load
[#1998](https://github.com/PrimeIntellect-ai/prime-rl/pull/1998) — Fix _tied_weights_keys for transformers v5 compat
[#1991](https://github.com/PrimeIntellect-ai/prime-rl/pull/1991) — Fix crash when pad_token_id is a list in model config
[#1986](https://github.com/PrimeIntellect-ai/prime-rl/pull/1986) — Patch get_encode_kwargs to prevent HF tokenizer left-truncation
[#1964](https://github.com/PrimeIntellect-ai/prime-rl/pull/1964) — Reduce orchestrator memory usage for VLM rollouts
[#1959](https://github.com/PrimeIntellect-ai/prime-rl/pull/1959) — Fix modality-aware batch distribution to prevent FSDP deadlocks
[#1955](https://github.com/PrimeIntellect-ai/prime-rl/pull/1955) — Disable timeout on admin httpx clients
[#1938](https://github.com/PrimeIntellect-ai/prime-rl/pull/1938) — Validate prompt length in TITO endpoint before computing max_tokens
[#1920](https://github.com/PrimeIntellect-ai/prime-rl/pull/1920) — Handle external CUDA_VISIBLE_DEVICES
[#1903](https://github.com/PrimeIntellect-ai/prime-rl/pull/1903) — Handle skipped eval intervals when ckpt_step jumps
[#1901](https://github.com/PrimeIntellect-ai/prime-rl/pull/1901) — Strictly enforce async level + fix off-policy off-by-one
[#1887](https://github.com/PrimeIntellect-ai/prime-rl/pull/1887) — Fix VLM cache
[#1875](https://github.com/PrimeIntellect-ai/prime-rl/pull/1875) — Per-component seq_len overrides shared seq_len
[#1862](https://github.com/PrimeIntellect-ai/prime-rl/pull/1862) — Fix effective_batch_size metric
[#1854](https://github.com/PrimeIntellect-ai/prime-rl/pull/1854) — Fix tool_call_parser not resolved when model name is propagated
[#1841](https://github.com/PrimeIntellect-ai/prime-rl/pull/1841) — Robustify env crash detection
[#1837](https://github.com/PrimeIntellect-ai/prime-rl/pull/1837) — Fix Hermes tool parser "Already borrowed" crash under concurrent load
[#1812](https://github.com/PrimeIntellect-ai/prime-rl/pull/1812) — Fix MoE weight conversion for vLLM compatibility
[#1808](https://github.com/PrimeIntellect-ai/prime-rl/pull/1808) — Fix missing param
[#1806](https://github.com/PrimeIntellect-ai/prime-rl/pull/1806) — Fix VLMs after transformers bump
[#1805](https://github.com/PrimeIntellect-ai/prime-rl/pull/1805) — Fix resume from latest step for single-run
[#1804](https://github.com/PrimeIntellect-ai/prime-rl/pull/1804) — Configure default logger instead of raise
[#1803](https://github.com/PrimeIntellect-ai/prime-rl/pull/1803) — Upgrade flash-attn from 2.6.3 to 2.8.3
[#1801](https://github.com/PrimeIntellect-ai/prime-rl/pull/1801) — Fix env server deadlock
[#1791](https://github.com/PrimeIntellect-ai/prime-rl/pull/1791) — Replace round robin
[#1781](https://github.com/PrimeIntellect-ai/prime-rl/pull/1781) — Fix custom token endpoint
[#1765](https://github.com/PrimeIntellect-ai/prime-rl/pull/1765) — Fix weight checkpoint saving for tied-embedding models
[#1764](https://github.com/PrimeIntellect-ai/prime-rl/pull/1764) — Fix off-policy tracker bug
[#1763](https://github.com/PrimeIntellect-ai/prime-rl/pull/1763) — Fix checkpoint cleanup crashing trainer on OSError
[#1762](https://github.com/PrimeIntellect-ai/prime-rl/pull/1762) — Fix checkpoint step ordering bug
[#1760](https://github.com/PrimeIntellect-ai/prime-rl/pull/1760) — Fix model weight loading: crash fix and memory optimization
[#1756](https://github.com/PrimeIntellect-ai/prime-rl/pull/1756) — Fix torch distributed warning by adding device_id handling
[#1754](https://github.com/PrimeIntellect-ai/prime-rl/pull/1754) — Fix Muon hparam passthrough
[#1752](https://github.com/PrimeIntellect-ai/prime-rl/pull/1752) — Fix eval metrics type not JSON serializable
[#1729](https://github.com/PrimeIntellect-ai/prime-rl/pull/1729) — Fix broadcast alpha rank

---

# Misc

[#2116](https://github.com/PrimeIntellect-ai/prime-rl/pull/2116) — Revamp docs and readme
[#1897](https://github.com/PrimeIntellect-ai/prime-rl/pull/1897) — Unify --dry-run flag across RL and SFT entrypoints
[#1872](https://github.com/PrimeIntellect-ai/prime-rl/pull/1872) — Add output_dir safety check to prevent accidental overwrites
[#1984](https://github.com/PrimeIntellect-ai/prime-rl/pull/1984) — Config warnings
[#1866](https://github.com/PrimeIntellect-ai/prime-rl/pull/1866) — Log reward statistics
[#1852](https://github.com/PrimeIntellect-ai/prime-rl/pull/1852) — Track seqs per rollout
[#2009](https://github.com/PrimeIntellect-ai/prime-rl/pull/2009) — Distinguish is_truncated and log stop_conditions
[#1966](https://github.com/PrimeIntellect-ai/prime-rl/pull/1966) — Add zero grad tracking
[#1835](https://github.com/PrimeIntellect-ai/prime-rl/pull/1835) — Add mismatch_kl-entropy ratio
[#1834](https://github.com/PrimeIntellect-ai/prime-rl/pull/1834) — Use lower-variance estimator for pass@k
[#2060](https://github.com/PrimeIntellect-ai/prime-rl/pull/2060) — Validate VLM dtype in configs
[#1993](https://github.com/PrimeIntellect-ai/prime-rl/pull/1993) — Move env packages to dedicated envs extras group
[#1778](https://github.com/PrimeIntellect-ai/prime-rl/pull/1778) — Set recompile limits
[#1684](https://github.com/PrimeIntellect-ai/prime-rl/pull/1684) — Multiple small sanity checks
[#1674](https://github.com/PrimeIntellect-ai/prime-rl/pull/1674) — Best-effort interleaving
[#1750](https://github.com/PrimeIntellect-ai/prime-rl/pull/1750) — Clean deps
[#1758](https://github.com/PrimeIntellect-ai/prime-rl/pull/1758) — Bump wandb to 0.24.2
[#1749](https://github.com/PrimeIntellect-ai/prime-rl/pull/1749) — Bump verifiers (bc9fcd5)
[#1744](https://github.com/PrimeIntellect-ai/prime-rl/pull/1744) — Update prime-rl to newest verifiers main
[#1780](https://github.com/PrimeIntellect-ai/prime-rl/pull/1780) — Bump verifiers v0.1.11.dev0
[#2036](https://github.com/PrimeIntellect-ai/prime-rl/pull/2036) — Bump verifiers
[#2051](https://github.com/PrimeIntellect-ai/prime-rl/pull/2051) — Bump verifiers to 1960e77
[#1733](https://github.com/PrimeIntellect-ai/prime-rl/pull/1733) — Add enable prefix caching arg passthrough to vLLM
[#1740](https://github.com/PrimeIntellect-ai/prime-rl/pull/1740) — Set VLLM_WORKER_MULTIPROC_METHOD=spawn by default
[#1747](https://github.com/PrimeIntellect-ai/prime-rl/pull/1747) — Add skills directory
[#1751](https://github.com/PrimeIntellect-ai/prime-rl/pull/1751) — Fix typos in docs
[#1620](https://github.com/PrimeIntellect-ai/prime-rl/pull/1620) — Use presigned URLs when uploading samples
[#1985](https://github.com/PrimeIntellect-ai/prime-rl/pull/1985) — Route env install output through structured logger
[#1775](https://github.com/PrimeIntellect-ai/prime-rl/pull/1775) — Make eval retries configurable and default to no retries
[#2017](https://github.com/PrimeIntellect-ai/prime-rl/pull/2017) — Make httpx connect timeout configurable
[#2012](https://github.com/PrimeIntellect-ai/prime-rl/pull/2012) — Allow num_infer_nodes=0 with fake data in multi-node SLURM RL
[#1759](https://github.com/PrimeIntellect-ai/prime-rl/pull/1759) — Add mini MoE

---

# Contributors

@mikasenghaas, @samsja, @hallerite, @JannikSt, @rasdani, @faresobeid, @S1ro1, @cdreetz, @sapiosaturn, @shayonj, @Jackmin801, @idoh, @eligotts, @philippnormann, @kalomaze, @manveerxyz, @leopold-tzafon, @snimu, @JohannesHa, @eexwhyzee, @crStiv, @d42me

