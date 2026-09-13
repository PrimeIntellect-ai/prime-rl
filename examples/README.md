# Examples

End-to-end usage examples for prime-rl, referenced from the top-level [README](../README.md).

- **[`basic/`](basic)** — walk-throughs for the core environments on 1–8 GPUs (baseline eval →
  optional SFT warmup → RL → eval), each with its own README:
  - `reverse-text` — smallest end-to-end loop (single-turn, 0.6B)
  - `alphabet-sort` — multi-turn, user simulator, LoRA
  - `wiki-search` — multi-turn tool calling, LoRA
  - `wordle` — multi-turn (~6-turn games)
  - `hendrycks-sanity` — single-turn math, long-running
  - `dynamo` — five-step Qwen3 math training with external Dynamo inference and NCCL weight updates
  - [`vlm/`](basic/vlm/README.md) — multimodal (VLM) SFT, dense + MoE LoRA configs
- **[`advanced/`](advanced)** — larger, mostly multi-node runs on frontier models, one folder
  per model, each with a launch README: [`qwen3-30b-a3b`](advanced/qwen3-30b-a3b/README.md)
  (math/swe/tool), [`glm-4.5-air`](advanced/glm-4.5-air/README.md) (search/swe/terminal),
  [`glm-5.2`](advanced/glm-5.2/README.md) (large-scale + PD-disaggregated inference),
  `intellect-3.1` (swe).

Frontier-model configs without launch walkthroughs (`minimax-m2.5`, `nemotron-3-super`,
`deepseek-v4-flash`) live in [`configs/advanced/`](../configs/advanced). Dev-sized (2-GPU)
counterparts of `basic/` live in [`configs/basic/`](../configs/basic).
