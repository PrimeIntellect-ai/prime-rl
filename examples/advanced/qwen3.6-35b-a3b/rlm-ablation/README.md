# Qwen3.6-35B-A3B with the rlm harness

Agentic SWE RL on `scaleswe` with the rlm harness at 128K. `base.toml` is the shared recipe: BF16 on trainer and engines, icepop with score centering, max staleness 8, batch 64 x group 8, 1 train + 1 infer node. The other files are overlays, one axis each:

| Axis | Overlay | Change |
|---|---|---|
| Hardware | `h200.toml` | Two inference replicas (H200, BF16) |
| Precision | `mxfp8.toml` | MXFP8 trainer and engines, fp8 KV cache (Blackwell) |
| | `nvfp4.toml` | MXFP8 dense linears and NVFP4 routed experts with 4/6 scaling, the trainer's forward matched to the engines (Blackwell) |
| Staleness | `stale.toml` | Weights to the engines every 64 steps, max staleness 64 |
| | `stale-nocap.toml` | Weights every 32 steps, no staleness cap, inference never pauses |
| Loss | `reinforce-kl.toml` | REINFORCE with score centering and a 0.05 sampler KL on stale tokens |
| | `reinforce-kl-prune.toml`, over `reinforce-kl.toml` | Zero-advantage rollouts pruned, so the KL acts only on rollouts with an advantage |
| | `reinforce-kl-mask.toml`, over `reinforce-kl.toml` | Token-level KL mask: drop tokens whose sampler-trainer KL exceeds 0.3 from the policy gradient and the KL |
| Format | `format-stop.toml` | Reward 0 at the first turn that breaks the Qwen3.6 tool-call format ([`format_violation.py`](format_violation.py)) |

Compose left to right from the repo root, e.g. NVFP4 with no staleness cap and the REINFORCE + KL loss:

```bash
uv run rl @ examples/advanced/qwen3.6-35b-a3b/rlm-ablation/base.toml @ examples/advanced/qwen3.6-35b-a3b/rlm-ablation/nvfp4.toml @ examples/advanced/qwen3.6-35b-a3b/rlm-ablation/stale-nocap.toml @ examples/advanced/qwen3.6-35b-a3b/rlm-ablation/reinforce-kl.toml
```

Rollouts run in [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview), with the code hosts blocked so a rollout can't fetch the task repo's upstream fix. Under staleness, `format-stop.toml` can cost more than it saves: a weight broadcast that breaks the tool-call format zeroes most rollouts until the next broadcast.
