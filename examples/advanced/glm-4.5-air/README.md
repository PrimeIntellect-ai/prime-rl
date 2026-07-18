# GLM-4.5-Air

Multi-node RL examples for `zai-org/GLM-4.5-Air` (rlm harness on the prime runtime).

| config | task |
|---|---|
| `search.toml`   | deep-research / search (openseeker, redsearcher, browsecomp) |
| `swe.toml`      | SWE (ScaleSWE + SWE-Bench-Verified) |
| `terminal.toml` | terminal (tmax + Terminal-Bench) |

```bash
uv run rl @ examples/advanced/glm-4.5-air/<task>.toml
```
