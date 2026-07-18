# Qwen3-30B-A3B

Multi-node RL examples for `Qwen/Qwen3-30B-A3B-Thinking-2507`.

| config | task |
|---|---|
| `math.toml` | math RL (i3-math, AIME eval) |
| `swe.toml`  | SWE (R2E-Gym + SWE-Bench-Verified) |
| `tool.toml` | general agentic tool use (general-agent) |

```bash
uv run rl @ examples/advanced/qwen3-30b-a3b/<task>.toml
```
