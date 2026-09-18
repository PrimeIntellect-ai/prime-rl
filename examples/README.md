# Examples

End-to-end usage examples for prime-rl, referenced from the top-level [README](../README.md).

New here? Follow [`basic/reverse-text/`](basic/reverse-text/README.md) first — it trains a small model end-to-end on 2 GPUs and introduces the launch → dashboard → checkpoint flow the advanced examples reuse.

## advanced/ — frontier models, multi-node

The advanced walkthrough explains how to compose an environment and a model-topology profile from
[`configs/advanced/`](../configs/advanced): [`advanced/README.md`](advanced/README.md). The model
folder also holds model-specific material such as [GLM-5.3 inference preflight notes](../configs/advanced/models/glm-5.3/infer.md).

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
- [`vlm/`](extra/vlm/README.md) — multimodal (VLM) SFT, dense + MoE LoRA configs

All runnable TOML files live under [`configs/`](../configs). Examples contain walkthroughs only.
