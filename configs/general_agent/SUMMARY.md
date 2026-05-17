# SFT — NemotronH-Nano-30B-A3B-Base on `general_agent` — v1

**Config**: [`sft_nemotron_nano_base.toml`](sft_nemotron_nano_base.toml)
**Wandb**: https://wandb.ai/primeintellect/general-agent/runs/nb89tj4m
**Output dir**: `/beegfs/mika/general-agent-sft-nemotron-nano-base`

## Config

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`, tokenizer from the Instruct ckpt
- Data: `PrimeIntellect/INTELLECT-5-SFT-Raw[general_agent]:train` (4,417 rows)
- seq_len 65,536 · batch_size 8 · max_steps 100 · lr 5e-5 (linear, warmup 25)
- Parallelism: 2 × H200 (16 GPUs), ep=2, cp=2, cp_style=ulysses
- Checkpoints every 20 steps

## Core stats

| step | loss | grad_norm | mfu | throughput |
|------|------|-----------|-----|------------|
| 1    | 0.70 | 3.23 | 40% | 43k tok/s |
| 10   | 0.45 | 1.08 | 56% | 60k tok/s |
| 50   | 0.32 | 0.37 | 62% | 66k tok/s |
| 99   | 0.19 | 0.23 | 75% | 80k tok/s |

~20 min wallclock · 0 NaN · 0 tokenization skips · peak mem 51 / 139.8 GiB (36.5%).

---

# v2 — lower LR, fewer steps, finer ckpts

Motivated by v1 bfcl-v3 results: peak at step_20 (0.621), monotonic collapse after (0.213 by step_100). Classic overfit on a 4.4k-row corpus. v2 takes the same data and just slows learning.

**Config**: [`sft_nemotron_nano_base_v2.toml`](sft_nemotron_nano_base_v2.toml)
**Wandb**: https://wandb.ai/primeintellect/general-agent/runs/z2hzt5qn
**Output dir**: `/beegfs/mika/general-agent-sft-nemotron-nano-base-v2`

## Diff from v1

| field | v1 | v2 |
|---|---|---|
| `lr` | 5e-5 | **1e-5** |
| `max_steps` | 100 | **40** |
| `warmup` | 25 | **10** (kept at 25% of max) |
| `ckpt interval` | 20 | **5** (8 ckpts: 5, 10, ..., 40) |

Everything else (model, data, batch=8, seq=64k, ep=2, cp=2, seed) identical.

## Core stats (live)

| step | loss | grad_norm | LR | throughput |
|------|------|-----------|-----|------------|
| 1    | 0.70 | 3.23 | 1.0e-6 | 42k tok/s |
| 5    | 0.65 | 1.98 | 5.0e-6 | 17k tok/s (ckpt save) |
| 10   | 0.53 | 1.19 | 1.0e-5 | 17k tok/s (ckpt save) |
| 15   | 0.43 | 1.00 | 1.0e-5 | 17k tok/s (ckpt save) |
| 20   | 0.48 | 1.03 | 1.0e-5 | 18k tok/s (ckpt save) |
| 25   | 0.36 | 0.80 | 1.0e-5 | 18k tok/s (ckpt save) |
| 30   | 0.39 | 0.91 | 1.0e-5 | 18k tok/s (ckpt save) |
| 35   | 0.40 | 0.92 | 4.4e-6 | 18k tok/s (ckpt save) |
| 39   | 0.38 | 1.01 | 1.0e-13 | 29k tok/s (final) |

(Throughput is ckpt-save-dragged at every milestone since `interval=5`.)

~16 min wallclock · 0 NaN · peak mem 51 / 139.8 GiB (36.5%). 8 ckpts saved: `step_{5,10,15,20,25,30,35,40}`.

---

# v2-p2 — same config + tools-fix

bfcl-v3 deep dive turned up a silent training/inference mismatch: prime-rl's SFT loader read tools from the `tools` column (single JSON string) but our dataset stores them under `tool_defs` (`list[str]` of OAI tool dicts). Result: chat template rendered an **empty system block** at training time (385 tokens / row), while vLLM injects the full tool defs + format example at inference (1598 tokens). Model never saw the inference-time context → over-prosed, multi-turn collapsed.

Fix landed in [prime-rl PR #2494](https://github.com/PrimeIntellect-ai/prime-rl/pull/2494): loader now accepts either column (`tools` or `tool_defs`) as a JSON string or `list[dict]`. Re-pushed the dataset with `tool_defs` as a single JSON-encoded string.

**Config**: [`sft_nemotron_nano_base_v2.toml`](sft_nemotron_nano_base_v2.toml) (unchanged; only the loader + dataset shape changed)
**Wandb**: https://wandb.ai/primeintellect/general-agent/runs/m8h5ebd5
**Output dir**: `/beegfs/mika/general-agent-sft-nemotron-nano-base-v2` (overwritten via `--clean-output-dir`)

## Core stats (live)

| step | loss | grad_norm | LR | throughput | vs v2 (no tools) |
|------|------|-----------|-----|------------|-------------------|
| 5    | 0.59 | 1.90 | 5.0e-6 | 18k tok/s | -0.06 |
| 10   | 0.47 | 1.22 | 1.0e-5 | 17k tok/s | -0.06 |
| 15   | 0.48 | 1.13 | 1.0e-5 | 18k tok/s | +0.05 |
| 20   | 0.41 | 0.86 | 1.0e-5 | 18k tok/s | -0.07 |
| 25   | 0.43 | 1.07 | 1.0e-5 | 18k tok/s | +0.07 |
| 30   | 0.33 | 0.74 | 1.0e-5 | 17k tok/s | -0.06 |
| 35   | 0.32 | 0.87 | 4.4e-6 | 17k tok/s | -0.08 |
| 39   | 0.38 | 0.97 | 1.0e-13 | 27k tok/s | 0.00 (final) |

~18 min wallclock · 0 NaN · peak mem 51 / 139.8 GiB (36.4%). 8 ckpts at `step_{5,10,15,20,25,30,35,40}` in `/beegfs/mika/general-agent-sft-nemotron-nano-base-v2/weights/`.

Loss not directly comparable across runs (the ~1200 extra system-prompt tokens shift the average-per-token loss baseline). Eval will be the deciding signal.

---

# Eval results

Tool-call benchmarks. Scores are the env's default reward metric (typically pass@1) at `num_examples=-1` unless noted.

| Model | `bfcl-v3` | `mcp-atlas` |
|-------|-----------|-------------|
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (vendor instruct) | **0.735** ([prime](https://app.primeintellect.ai/dashboard/evaluations/jay9quphy3t9zmofi0jr6nd0)) | **0.455** ([prime](https://app.primeintellect.ai/dashboard/evaluations/k8muk3n0hxhadf5ww64lyjxw)) |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` (vendor base, `max_tokens=4096`) | **0.189** ([prime](https://app.primeintellect.ai/dashboard/evaluations/gsoczduu35xzu0mmvwl3asv1)) | **0.006** ([prime](https://app.primeintellect.ai/dashboard/evaluations/d9dfd6vw9sfjdcsth5t5r42k)) |
| `…sft-nemotron-nano-base-v2/weights/step_40/` (v2-p3) | **0.210** ([prime](https://app.primeintellect.ai/dashboard/evaluations/sib2v6h1e1ndv7iftdv1abl6)) | **0.015** ([prime](https://app.primeintellect.ai/dashboard/evaluations/ny237zwrkutys7llwlrb6pcn)) |
| `…sft-nemotron-nano-base-v2/weights/step_80/` (v2-p3) | **0.479** ([prime](https://app.primeintellect.ai/dashboard/evaluations/aoc6byzlgwafftncncvb82qo)) | **0.047** ([prime](https://app.primeintellect.ai/dashboard/evaluations/duqk9np6o9lumduk7lnecxp2)) |
| `…sft-nemotron-nano-base-v2/weights/step_120/` (v2-p3) | **0.494** ([prime](https://app.primeintellect.ai/dashboard/evaluations/bu33rae9bx0el13e1urpyfgx)) | **0.053** ([prime](https://app.primeintellect.ai/dashboard/evaluations/xu76vk84drsst5wjh2xoh20a)) |
| `…sft-nemotron-nano-base-v2/weights/step_160/` (v2-p3) | **0.517** ([prime](https://app.primeintellect.ai/dashboard/evaluations/jwy6erzlv45r1aq08k29tp3v)) | **0.121** ([prime](https://app.primeintellect.ai/dashboard/evaluations/wrplaqn07eth2icjscqb3gex)) |
| `…sft-nemotron-nano-base-v2/weights/step_200/` (v2-p3) | **0.523** ([prime](https://app.primeintellect.ai/dashboard/evaluations/enpo0upme0c5hw2882tdmvvz)) | **0.089** ([prime](https://app.primeintellect.ai/dashboard/evaluations/ub3p6il78xifxhd3gxfvpi40)) |

## Eval setup

- `bfcl-v3`: `num_examples=-1`, `rollouts_per_example=1`, `max_tokens=131072`, `max_concurrent=128`
- `mcp-atlas`: `rollouts_per_example=4`
