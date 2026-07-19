# reverse_text (2 GPU)

Smallest end-to-end RL loop — single-turn, `Qwen3-0.6B`. Dev mirror of
[`examples/basic/reverse-text`](../../../examples/basic/reverse-text).

```bash
uv run sft @ configs/basic/reverse-text/sft.toml   # optional warmup
uv run rl  @ configs/basic/reverse-text/rl.toml
```
