# TTT (test-time training) experiment configs

Train a per-rollout LoRA at every context compaction (the `compacting` harness). Design,
constraints, and managed-launch contracts: [`docs/ttt.md`](../../docs/ttt.md).

## Processes

Every TTT arm needs three processes (eval) or four (RL):

1. **Inference** — vLLM with LoRA serving: `uv run inference @ configs/ttt/inference.toml`
2. **TTT service** — the per-rollout adapter trainer (own GPU): `uv run ttt @ configs/ttt/ttt.toml`
3. **Eval** — the v1 eval CLI, from `deps/verifiers` (or an installed env): see below.
4. *(RL only)* trainer + orchestrator via `uv run rl @ configs/ttt/rl_compaction_ttt.toml`.

The TTT service and the inference server must share a filesystem (`output_dir`) and agree
on the base model. The eval must use the **train client** (`--client.type train`) — TTT
consumes exact token ids.

## Launch commands

Eval arms (deepdive), common flags:

```bash
CLIENT="--client.type train --client.base-url http://localhost:8000/v1 --model <MODEL>"
LIMITS="--max-input-tokens 32768"   # total-context budget for every arm
```

| Arm | Command sketch |
|---|---|
| full context | `uv run eval deepdive-v1 --harness.search true $CLIENT $LIMITS` |
| small plain context | `uv run eval deepdive-v1 --harness.search true $CLIENT --max-input-tokens 8192` |
| compaction only | `uv run eval deepdive-v1 --harness.id compacting --harness.search true --harness.compact-at-tokens 8192 $CLIENT $LIMITS` |
| compaction + TTT | ... `--ttt.base-url http://localhost:8092` |
| compaction only through the TTT-configured path (hook disabled) | ... `--ttt.base-url ... --ttt.enabled false` |
| compaction + Q&A-TTT | ... `--ttt.base-url ... --ttt.qa.num-generations 2 --ttt.qa.items-per-generation 4` |

RL (small scale): `uv run rl @ configs/ttt/rl_compaction_ttt.toml` (start the TTT service first).

RL experiment family (scaleswe, GLM-4.5-Air) — the A0–A5 arm matrix, launcher-managed
TTT service (see `scaleswe/base.toml` header):

```bash
uv run rl @ configs/ttt/scaleswe/base.toml @ configs/ttt/scaleswe/arm_a<0|1>_*.toml                                     # baselines
uv run rl @ configs/ttt/scaleswe/base.toml @ configs/ttt/scaleswe/ttt_common.toml @ configs/ttt/scaleswe/arm_a<2-5>_*.toml  # TTT arms
```
