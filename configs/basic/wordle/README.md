# wordle (2 GPU)

Multi-turn Wordle (~6-turn games), `Qwen3-1.7B-Wordle-SFT`. Dev mirror of
[`examples/basic/wordle`](../../../examples/basic/wordle).

```bash
uv run sft @ configs/basic/wordle/sft.toml   # optional warmup
uv run rl  @ configs/basic/wordle/rl.toml
```
