# Vision-language training

Multimodal RL and SFT configurations using the custom [PrimeRL VLM implementations](../../../docs/advanced.md#multimodal-training). The RL templates keep the model and training settings from our long-context browser runs, with placeholders for your environments, run names and paths.

## Image RL

| Config | Model | Batch / group | Hardware |
|---|---|---|---|
| [`rl-qwen3.5.toml`](rl-qwen3.5.toml) | `Qwen/Qwen3.5-122B-A10B` | 336 / 16 | 4 trainer + 4 inference nodes, 8× H200 141 GB per node |
| [`rl-qwen3.8.toml`](rl-qwen3.8.toml) | `Qwen/Qwen3.8-Flash-Next` | 256 / 16 | same |

Both use Muon at `1e-6`, optimizer-state CPU offload, FP32 optimizer state and gradient reduction, full activation checkpointing with offload, CP4 / EP8, and a frozen vision encoder. Training uses GRPO with IPO (`eps = 0.3`), binary rewards and seeded shuffled task selection for 200 steps. The batch is the number of rollout episodes, so these are 21 and 16 groups per step respectively.

The sequence limit is 237,568 tokens, with browser-harness compaction at 184,320 tokens and at most 200 turns. Per-turn generation allows 16,384 tokens at temperature 1 with no top-p/top-k truncation. Episode concurrency is 96 initial / 400 maximum. Held-out eval runs at step 0 and every 10 steps, with two attempts per task; checkpoints are saved every 10 steps, keeping the latest one.

Inference has four independent one-node replicas, each using TP4 / DP2 with expert parallelism. It uses BF16 serving, FP32 LM-head and router logits, prefix caching, `mm_encoder_tp_mode = "data"`, and sticky least-loaded routing. Qwen3.8 also offloads its ngram tables to CPU and uses the `qwen3.8` renderer with preserved thinking and `reasoning_effort = "xhigh"`; Qwen3.5 uses its own `qwen3.5` renderer. These settings are starting points for the stated hardware, not a guarantee that every task or context distribution fits.

### Customize before launching

- Replace `your-org/your-favorite-env@0.1.0` and `your-org/your-held-out-env@0.1.0` with versioned, image-capable tasksets and compatible harnesses. They are placeholders, not published environments. Install or fetch those environments before validating the config.
- `reward_mode`, `max_turns`, `max_request_bytes`, `context_limit_tokens`, `compaction_trigger_tokens` and `history_trimming_enabled` are browser task/harness options, not universal Verifiers settings. Keep them if your environment supports them, otherwise adapt them to its schema. Other harnesses need their own compaction implementation; `seq_len` alone does not enable compaction. The training request limit here is 60,000,000 bytes.
- The templates use one training source with all the sampling weight. To train a mixture, add sources with distinct names and explicit `ratio` values. No private catalog IDs, task mix, evaluation set or run IDs are embedded.
- Replace `your-run-name` in `[run]`, `[slurm]` and `[monitors.wandb]`, and set your W&B entity/project. Leave W&B IDs unset for a new run. To resume, keep the same run name and output directory and use `--resume` or `--resume.step N`.
- Put the checkout and outputs on a shared filesystem, replacing `/shared/outputs` as needed. Set your Slurm partition/account using the [launcher settings](../../../docs/scaling.md). The topology allocates eight nodes; the orchestrator shares an inference node.
- Export `WANDB_API_KEY`, your sandbox credentials (for the `prime` runtime), and any browser or judge keys required by your environment. Use your own team context. Do not put secrets in the TOML.

Install the GPU dependencies and kernels as described in the [root README](../../../README.md#setup). From the shared checkout, validate without submitting:

```bash
uv run rl @ examples/extra/vlm/rl-qwen3.8.toml --dry-run
```

After replacing the placeholders and reviewing the generated component configs and Slurm script, launch by omitting `--dry-run`. Use `rl-qwen3.5.toml` for the other model. Monitor with W&B or `uv run dashboard /shared/outputs`; check `mismatch_kl/all/mean`, `masked_mismatch_kl/mean`, `unmasked_mismatch_kl/mean` and `is_masked/mean` alongside reward and memory. The Qwen3.8 trainer example supports still images; video inputs are not supported.

## SFT

The SFT configs use the Qwen3.5 dense and Qwen3.6 MoE VLM implementations. Both LoRA-finetune with a frozen vision encoder and assistant-only loss masking.

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
