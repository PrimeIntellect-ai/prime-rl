# TTT (test-time training) experiment configs

The compaction-time-training experiments from `ttt-plan.md` / `ttt-implementation.md`:
train a per-rollout LoRA at every context compaction (the `compacting` harness), so what
falls out of the attention window lives on in the weights.

All numbers here (context budgets, thresholds, ranks, LRs) are illustrative — tune per
model/GPU. Environment: `deepdive_v1` (deep-research QA, needs a search-capable harness:
the compacting/default harness with `search = true` + `SERPER_API_KEY`).

## Processes

Every TTT arm needs three processes (eval) or four (RL):

1. **Inference** — vLLM with LoRA serving:
   `uv run inference @ configs/ttt/inference.toml`
2. **TTT service** — the per-rollout adapter trainer (own GPU):
   `uv run ttt @ configs/ttt/ttt.toml`
3. **Eval** — the v1 eval CLI, from `deps/verifiers` (or an installed env):
   see the arm commands below.
4. *(RL only)* trainer + orchestrator via `uv run rl @ configs/ttt/rl_compaction_ttt.toml`.

The TTT service and the inference server must share a filesystem (`output_dir`) and agree
on the base model. The eval must use the **train client** (`--client.type train`) — TTT
consumes exact token ids.

## Eval arms

Common flags (fill in the model/endpoint):

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

## RL arm (small scale)

`rl_compaction_ttt.toml` — GRPO on deepdive with compaction+TTT rollouts and frozen-adapter
replay in the trainer (start the TTT service first). Constraints enforced by config
validation: full-weight policy training (no `[trainer.model.lora]`), `enable_lora = true`
on inference. For the Q&A→policy arm, edit the env's `ttt` line in the TOML to
`ttt = { base_url = "...", qa = { recycle_to_policy = true } }` (or apply an overlay TOML
with the full `[[orchestrator.train.env]]` block) — dotted CLI overrides cannot index into
the env list.

## RL experiment family (scaleswe, GLM-4.5-Air)

`scaleswe/` — the A0–A5 arm matrix on `scaleswe-v1` (see the header of
`scaleswe/base.toml`): pure-RL and compaction baselines, compaction+TTT, QA, and the two
auxiliary-CE variants (naive recycle vs group meta-lessons, head-to-head). Their CE samples
share the RL batch and optimizer step; they are not separate follow-up SFT steps. TTT arms use
the **fsdp engine**: the prime-rl trainer stack with `max_slots` resident MultiLoRA
adapter slots and cross-rollout batched updates. The service is **launcher-managed** —
each TTT arm sets `deployment.num_ttt_nodes = 1` plus a `[ttt]` section, and the SLURM
launcher allocates the node, torchruns the service, and auto-wires output_dir / model /
vLLM admin roots / env `base_url`s (see `docs/ttt.md`). `scaleswe/ttt_service.toml`
remains for running the service standalone. `[engine] type = "peft"` (the default) is the
single-GPU engine for small models.

Every arm overlay has its own output directory, SLURM job name, and Prime sandbox label, so
the six supplied configs do not overwrite one another. This is naming discipline, not a
global launch reservation: two concurrent submissions of the same arm can still target the
same directory. Use a distinct output override for concurrent/repeated runs. The base does
not opt into checkpoint resume; pass `--ckpt.resume-step` only when intentionally resuming
that arm's existing directory.

Every arm pins `AweAI-Team/Scale-SWE` to commit
`d8db20390a936bbda9c96d88b97cc4778dff1481` and sets
`filter_unavailable_images = false`. Each arm also sets `use_prime_registry = true`: the
public mirror is known to lack some ScaleSWE image tags, while the Prime registry is the
complete source used by these sandbox runs. Dataset rows, task order, and image references
are therefore identical across the panel, and an unavailable image fails its task instead
of silently removing it from one arm. The dataset stores Docker tags rather than immutable
image digests; archive the registry-resolved image digest with run artifacts if binary-level
container provenance is required.

TTT-enabled eval environments use the renderer client because exact token IDs are required.
A0 and A1 retain Prime-RL's normal non-TTT eval transport. That transport difference should
be recorded as a limitation when comparing the baseline arms with A2–A5; a generic
renderer-client override was intentionally not added to Prime-RL core on this branch.
