# VLM examples

## Image RL

| Config | Model | Batch / group | CP |
|---|---|---|---|
| [`rl-qwen3.8.toml`](rl-qwen3.8.toml) | `Qwen/Qwen3.8-Flash-Next` | 256 / 16 | 8 |

The Qwen3.8 example targets **4 trainer + 4 inference nodes**, each with 8× H200 GPUs.

Replace the environment/harness IDs, run and W&B names, and shared output path. Install your environments and export the credentials they require. Reward and compaction fields are browser-environment options; adapt them to your harness's schema.

After [setup](../../../README.md#setup), validate from a shared checkout:

```bash
uv run rl @ examples/extra/vlm/rl-qwen3.8.toml --dry-run
```

Remove `--dry-run` to submit. See [Scaling](../../../docs/scaling.md) for Slurm configuration and [Training](../../../docs/training.md) for monitoring and resume.

## SFT

Multimodal (VLM) supervised fine-tuning configs, dense and MoE. VLM training (`[model.vlm]` set) requires a PrimeRL VLM implementation (see [Advanced § Multimodal Training](../../../docs/advanced.md#multimodal-training)); these configs use the Qwen3.5 dense and Qwen3.6 MoE VLM implementations. Both LoRA-finetune with a frozen vision encoder and assistant-only loss masking.

| Config | Model | Hardware |
|---|---|---|
| [`sft.toml`](sft.toml) | `Qwen/Qwen3.5-0.8B` (dense, LoRA rank 16) | tiny — sanity-scale (20 steps) |
| [`sft-moe.toml`](sft-moe.toml) | `Qwen/Qwen3.6-35B-A3B` (MoE, LoRA rank 32, `ep = 8`) | one 8-GPU node (`num_train_gpus = 8`) |

### Run it

Point `data.name` at your multimodal chat dataset (HF dataset with a `messages` column — see [Training § Dataset Format](../../../docs/training.md#dataset-format)), then:

```bash
uv run sft @ examples/extra/vlm/sft.toml
```

Notes:

- The MoE config declares `[deployment]` with `num_train_gpus = 8`; keep its `ep` in sync with the GPU count.
- `micro_batch_size = 1` is required for VLM SFT — do not raise it.
- Checkpoints land in `<run_dir>/checkpoints/step_<N>`; export HF-format weights with `tools/convert_dcp_to_bf16.py` (see [Training](../../../docs/training.md)).
