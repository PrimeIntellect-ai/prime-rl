# Examples

End-to-end usage examples for prime-rl, referenced from the top-level [README](../README.md).

- **[`basic/`](basic)** — walk-throughs for the core environments on 2 GPUs (baseline eval →
  optional SFT warmup → RL → eval), each with its own README:
  - `reverse_text` — smallest end-to-end loop (single-turn, 0.6B)
  - `alphabet_sort` — multi-turn, user simulator, LoRA
  - `wiki_search` — multi-turn tool calling, LoRA
  - `wordle` — multi-turn (~6-turn games)
  - `hendrycks_sanity` — single-turn math, long-running
- **[`advanced/`](advanced)** — larger, mostly multi-node runs on frontier models, one folder
  per model: `qwen3-30b-a3b` (math/swe/tool), `glm-4.5-air` (search/swe/terminal), `glm-5.2`
  (large-scale + PD-disaggregated inference), `minimax-m2.5` (swe), `nemotron-3-super` (swe),
  `intellect-3.1` (swe).

Dev-sized (2-GPU) counterparts of `basic/` live in [`configs/basic/`](../configs/basic).
