# PrimeRL Stress Matrix

These overlays run 10 steps of the real `alphabet_sort` RL example through
`gpu_layout` on the current 8-node allocation with 4 GPUs per node. Use them on
top of `examples/alphabet_sort/rl.toml`.

Topology variants:

- `2x4`: one trainer node and one inference node.
- `4x4`: one trainer node and three inference nodes.
- `8x4`: one trainer node and seven inference nodes.

Renderer selection is explicit: Qwen3 uses `qwen3`, OLMo3 uses `olmo3`,
GPT-OSS uses `gpt-oss`, and Gemma4 uses `gemma4`.

Gemma4 coverage includes `google/gemma-4-E2B-it` on 2x4/4x4/8x4 plus
representative family shapes on 2x4 or 4x4:

- `google/gemma-4-E4B-it`
- `google/gemma-4-31B-it`
- `google/gemma-4-26B-A4B-it`

Example:

```bash
env -u VIRTUAL_ENV WANDB_MODE=disabled uv run --no-sync rl \
  @ examples/alphabet_sort/rl.toml \
  @ examples/stress_matrix/qwen3_alphabet_sort_8x4_gpu_layout.toml \
  --no-wandb \
  --no-ckpt
```
