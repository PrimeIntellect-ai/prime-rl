# VLM SFT

Multimodal (VLM) supervised fine-tuning configs, dense and MoE. VLM training requires a custom PrimeRL implementation (`impl = "custom"`, `[model.vlm]` set — see [Advanced § Multimodal Training](../../../docs/advanced.md#multimodal-training)); these configs use the Qwen3.5 dense and Qwen3.6 MoE VLM implementations. Both LoRA-finetune with a frozen vision encoder and assistant-only loss masking.

| Config | Model | Hardware |
|---|---|---|
| [`sft.toml`](sft.toml) | `Qwen/Qwen3.5-0.8B` (dense, LoRA rank 16) | tiny — sanity-scale (20 steps) |
| [`sft-moe.toml`](sft-moe.toml) | `Qwen/Qwen3.6-35B-A3B` (MoE, LoRA rank 32, `ep = 8`) | one 8-GPU node (`num_train_gpus = 8`) |

## Run it

Point `data.name` at your multimodal chat dataset (HF dataset with a `messages` column — see [Training § Dataset Format](../../../docs/training.md#dataset-format)), then:

```bash
uv run sft @ examples/extra/vlm/sft.toml
```

Notes:

- The MoE config declares `[deployment]` with `num_train_gpus = 8`; keep its `ep` in sync with the GPU count.
- `micro_batch_size = 1` is required for VLM SFT — do not raise it.
- Checkpoints land in `<run_dir>/checkpoints/step_<N>`; export HF-format weights with `tools/convert_dcp_to_bf16.py` (see [Training](../../../docs/training.md)).
