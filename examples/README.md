# Examples

End-to-end usage examples for prime-rl, referenced from the top-level [README](../README.md).

New here? Follow [`basic/reverse-text/`](basic/reverse-text/README.md) first — it trains a small model end-to-end on 2 GPUs and introduces the launch → dashboard → checkpoint flow the advanced examples reuse.

## advanced/ — frontier models, multi-node

Larger runs on frontier models, one folder per model, each with a launch README covering requirements, credentials, launch, monitoring, and resume:

- [`qwen3-30b-a3b/`](advanced/qwen3-30b-a3b/README.md) — `Qwen3-30B-A3B` RL on math (`i3_math`), SWE (`r2e-gym`), and agentic tool use (general-agent on Modal). 2 train + 2 infer nodes; the SWE config runs at 131k context with `ep = 8` + `cp = 2`.
- [`glm-4.5-air/`](advanced/glm-4.5-air/README.md) — `GLM-4.5-Air` (100B MoE) RL across search, SWE, and terminal domains at 131k context, with agents in sandboxes. 2 train + 4 infer nodes (SWE: 1 + 3).
- [`glm-5.3/`](advanced/glm-5.3/README.md) — the GLM-5 family at scale: 16 trainer nodes, P/D-disaggregated FP8 inference, and an llm-d router + Mooncake KV-offload variant. Also ships standalone `infer/` pre-flights.
- `inkling2/` — `Inkling 2 Small` (594B MoE) math and SWE RL on H200 (`math-rl.toml` at 8k, `swe-rl.toml` at 16k): 6 train + 2 infer nodes, FP8 MoE experts on both sides, sign-SGD with full CPU offload.
- `intellect-3.1/` — reproduce the `INTELLECT-3.1` run: 4 train + 12 infer nodes on SWE.
- [`qwen3-235b-a22b/`](advanced/qwen3-235b-a22b/README.md) — SFT max trainer perf: `Qwen3-235B-A22B` on 4 trainer nodes at 16k context, a throughput reference using the max trainer throughput recipe.

Per-model RL configs at 256k context in three layers (`<model>/rl.toml` + `<model>/gpu/h200_<N>nodes.toml` + one of `data/scaleswe.toml`, `data/scaleswe-short.toml`, `data/math.toml`) for Muse Glimmer 30B, Qwen3.8 27B, Qwen3.8 Flash Next, GLM-5.3 Flash and Nemotron 3 Super. Qwen3.6-35B-A3B, DeepSeek-V4-Flash (also on B300), Nemotron 3.5 Super and Inkling 2 Small (16k) are in the same layout but not tested in it yet. See [`advanced/README.md`](advanced/README.md) for the trainer fits.

All advanced configs submit through SLURM (the `rl` entrypoint writes the sbatch job when the config has a `[slurm]` table) and are monitored with `uv run dashboard`. See the per-model README for the full walkthrough.

## basic/ — 1 to 8 GPUs

Walk-throughs for the core environments (baseline eval → optional SFT warmup → RL → eval), each with its own README:

- [`reverse-text/`](basic/reverse-text/README.md) — smallest end-to-end loop (single-turn, 0.6B): `eval.toml` → `sft.toml` → `rl.toml`
- [`alphabet-sort/`](basic/alphabet-sort/README.md) — multi-turn, user simulator, LoRA
- [`wiki-search/`](basic/wiki-search/README.md) — multi-turn tool calling, LoRA
- [`wordle/`](basic/wordle/README.md) — multi-turn (~6-turn games)
- [`hendrycks-sanity/`](basic/hendrycks-sanity/README.md) — single-turn math, long-running

## extra/ — beyond the core loop

Examples that don't follow the basic eval → SFT → RL walk-through pattern:

- [`dynamo/`](extra/dynamo/README.md) — five-step Qwen3 math training with external Dynamo inference and NCCL weight updates
- [`vlm/`](extra/vlm/README.md) — Qwen3.5 / Qwen3.8 multi-node image RL and dense + MoE LoRA SFT

## Related config folders

- Frontier-model configs without launch walkthroughs (`minimax-m2.5`, `nemotron-3-super`, `deepseek-v4-flash` SFT and serving) live in [`configs/advanced/`](../configs/advanced).
