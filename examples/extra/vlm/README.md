# Vision-language training

RL and SFT examples for the custom [PrimeRL VLM implementations](../../../docs/advanced.md#multimodal-training).

## Image RL

| Config | Model | Batch / group | CP |
|---|---|---|---|
| [`rl-qwen3.5.toml`](rl-qwen3.5.toml) | `Qwen/Qwen3.5-122B-A10B` | 336 / 16 | 4 |
| [`rl-qwen3.8.toml`](rl-qwen3.8.toml) | `Qwen/Qwen3.8-Flash-Next` | 256 / 16 | 8 |

Both target **4 trainer + 4 inference nodes**, each with 8× H200 GPUs.

Replace the environment/harness IDs, run and W&B names, and shared output path. Install your environments and export the credentials they require. Reward and compaction fields are browser-environment options; adapt them to your harness's schema.

After [setup](../../../README.md#setup), validate from a shared checkout:

```bash
uv run rl @ examples/extra/vlm/rl-qwen3.8.toml --dry-run
```

Remove `--dry-run` to submit; swap in `rl-qwen3.5.toml` for Qwen3.5. See [Scaling](../../../docs/scaling.md) for Slurm configuration and [Training](../../../docs/training.md) for monitoring and resume.

## SFT

Both configs use LoRA with a frozen vision encoder and assistant-only loss masking.

| Config | Model | Hardware |
|---|---|---|
| [`sft.toml`](sft.toml) | `Qwen/Qwen3.5-0.8B` (dense, LoRA rank 16) | sanity-scale, 20 steps |
| [`sft-moe.toml`](sft-moe.toml) | `Qwen/Qwen3.6-35B-A3B` (MoE, LoRA rank 32, EP8) | one 8-GPU node |

Set `data.name` to your [multimodal chat dataset](../../../docs/training.md#dataset-format), then:

```bash
uv run sft @ examples/extra/vlm/sft.toml
```

Keep `micro_batch_size = 1` for VLM SFT and the MoE config's EP degree aligned with its GPU count. See [Training](../../../docs/training.md) for checkpoint export.
