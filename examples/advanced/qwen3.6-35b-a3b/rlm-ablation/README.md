# Qwen3.6-35B-A3B with the rlm harness

Agentic SWE RL on `scaleswe` with the rlm harness at 128K. `base.toml` is the shared recipe: BF16 on trainer and engines, icepop with score centering, max staleness 8, batch 64 x group 8, 1 train + 1 infer node. The other files are overlays for hardware and precision:

| Overlay | Change |
|---|---|
| `h200.toml` | Two inference replicas (H200, BF16) |
| `mxfp8.toml` | MXFP8 trainer and engines, fp8 KV cache (Blackwell) |
| `nvfp4.toml` | MXFP8 dense linears and NVFP4 routed experts with 4/6 scaling, the trainer's forward matched to the engines (Blackwell) |

Compose from the repo root, e.g. the NVFP4 arm:

```bash
uv run rl @ examples/advanced/qwen3.6-35b-a3b/rlm-ablation/base.toml @ examples/advanced/qwen3.6-35b-a3b/rlm-ablation/nvfp4.toml
```

Rollouts run in [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview), with the code hosts blocked so a rollout can't fetch the task repo's upstream fix.
