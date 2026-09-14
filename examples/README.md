# Examples

End-to-end usage examples for prime-rl, referenced from the top-level [README](../README.md).

New here? Follow [`basic/reverse-text/`](basic/reverse-text/README.md) first — it trains a small model end-to-end on 2 GPUs and introduces the launch → dashboard → checkpoint flow the advanced examples reuse.

## advanced/ — frontier models, multi-node

Larger runs on frontier models, one folder per model, each with a launch README covering requirements, credentials, launch, monitoring, and resume:

- [`qwen3-30b-a3b/`](advanced/qwen3-30b-a3b/README.md) — `Qwen3-30B-A3B` RL on math (`i3_math`), SWE (`r2e-gym`), and agentic tool use (general-agent on Modal). 2 train + 2 infer nodes; the SWE config runs at 131k context with `ep = 8` + `cp = 2`.
- [`glm-4.5-air/`](advanced/glm-4.5-air/README.md) — `GLM-4.5-Air` (100B MoE) RL across search, SWE, and terminal domains at 131k context, with agents in sandboxes. 2 train + 4 infer nodes (SWE: 1 + 3).
- [`glm-5.2/`](advanced/glm-5.2/README.md) — the GLM-5 family at scale: 16 trainer nodes, P/D-disaggregated FP8 inference, and an llm-d router + Mooncake KV-offload variant. Also ships standalone `infer/` pre-flights.
- `intellect-3.1/` — reproduce the `INTELLECT-3.1` run: 4 train + 12 infer nodes on SWE.

All advanced configs submit through SLURM (the `rl` entrypoint writes the sbatch job when the config has a `[slurm]` table) and are monitored with `uv run dashboard`. See the per-model README for the full walkthrough.

## basic/ — 1 to 8 GPUs

Walk-throughs for the core environments (baseline eval → optional SFT warmup → RL → eval), each with its own README:

- [`reverse-text/`](basic/reverse-text/README.md) — smallest end-to-end loop (single-turn, 0.6B)
- [`alphabet-sort/`](basic/alphabet-sort/README.md) — multi-turn, user simulator, LoRA
- [`wiki-search/`](basic/wiki-search/README.md) — multi-turn tool calling, LoRA
- [`wordle/`](basic/wordle/README.md) — multi-turn (~6-turn games)
- [`hendrycks-sanity/`](basic/hendrycks-sanity/README.md) — single-turn math, long-running

## extra/ — beyond the core loop

Examples that don't follow the basic eval → SFT → RL walk-through pattern:

- [`dynamo/`](extra/dynamo/README.md) — five-step Qwen3 math training with external Dynamo inference and NCCL weight updates
- [`vlm/`](extra/vlm/README.md) — multimodal (VLM) SFT, dense + MoE LoRA configs

## Related config folders

- Frontier-model configs without launch walkthroughs (`minimax-m2.5`, `nemotron-3-super`, `deepseek-v4-flash`) live in [`configs/advanced/`](../configs/advanced).
- Standalone eval configs live in [`configs/evals/`](../configs/evals).
- Dev-sized (2-GPU) counterparts of `basic/` live in [`configs/basic/`](../configs/basic).
