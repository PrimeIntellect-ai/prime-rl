# Test-Time Training (TTT)

TTT trains a **per-rollout LoRA adapter during the rollout itself**: every time the agent's
context is compacted (the `compacting` harness), the just-dropped branch is trained into the
adapter, and the rest of the rollout samples through it — so what fell out of the attention
window lives on in the weights. For RL, every branch is replayed in the trainer under the
exact adapter version it was sampled with, so the importance ratio stays honest: the adapter
acts as context, exactly like the tokens do.

Plan and design rationale: `ttt-plan.md` / `ttt-implementation.md` (repo root).
Experiment configs: [`configs/ttt/`](../configs/ttt/README.md).

## Architecture

Four processes (the standard three plus the TTT service):

```
harness (compacting)          interception (verifiers.v1)         TTT service (ttt)
  rewrites context     →   fork detected in the trace graph   →   POST /update
                                                                   (grad steps on the
  next turn samples    ←   model = adapter, cache salted      ←    branch's exact tokens,
  through the adapter                                              PEFT ckpt, vLLM
                                                                   /load_lora_adapter)
```

- **The invariant**: every trace branch is sampled under exactly one adapter version.
  Updates fire only at branch forks (compactions); each committed node is stamped with its
  version (`MessageNode.ttt_version`); `Branch.ttt_version` enforces uniformity.
- **The trigger is passive**: the interception layer detects that a prepared turn doesn't
  extend the previous leaf. Harnesses know nothing about TTT.
- **Exactness over latency**: the rollout blocks on each update (applied + adapter reloaded
  + prefix cache salted per version) before the next turn samples.
- **TTT requires the renderer (train) client** — updates consume the engine's exact token
  ids; the eval relay fails loudly.

## Configuration

Env-side (`verifiers.v1.TTTConfig`, on any v1 env / the eval CLI as `--ttt.*`):

```toml
[[orchestrator.train.env]]
taskset = { id = "deepdive-v1" }
harness = { id = "compacting", compact_at_tokens = 8192 }
ttt = { base_url = "http://localhost:8092" }          # the TTT service
# loss_scope = "all" | "sampled"; train_final_branch; qa = {...} (below)
```

Service-side (`TTTServiceConfig`, `uv run ttt @ config.toml`): base model (must match the
inference deployment), LoRA rank/alpha/targets, optimizer + LR + `steps_per_update`,
`inference_admin_urls` (every vLLM server), `output_dir` (checkpoints at
`outputs/ttt/<rollout_id>/v<k>/`, shared FS with inference and trainer).

Inference: `enable_lora = true`, `max_lora_rank >=` the TTT rank, `max_loras >=` concurrent
TTT rollouts.

## Q&A at compaction (Cartridges-style)

`ttt.qa = { num_pairs = 8 }` generates question–answer pairs about the abandoned branch
*with the branch still in context* (through the rollout's own client, under its current
adapter), then trains the adapter on the pairs rendered **standalone** — the knowledge must
come from the weights, not a context-conditioned mapping. `qa.also_train_rollout` adds the
raw branch back into the update. Q&A exchanges are framework housekeeping: never on the
trace, never counted against `RolloutLimits` (own `qa.max_tokens` budget), recorded in
`trace.info["ttt"]`.

`qa.recycle_to_policy = true` (RL) additionally recycles the pairs into the **policy's**
main weights: the orchestrator renders them with the policy tokenizer and routes them to the
`ce` loss component, riding the same training batch as the RL samples.

## RL replay

- `TrainingSample.ttt_adapter_path` carries each branch's adapter checkpoint ref
  (resolved from the trace stamps; a stamped branch with no recorded checkpoint is a hard
  error, never a silent base-model replay).
- The packer never mixes adapter paths in a micro batch; the trainer's `TTTReplayManager`
  applies the frozen adapter around each micro batch's forward via forward hooks (no
  parameters, composes with FSDP/AC/compile; gradients flow to base weights only).
- Constraints (validated at launch): full-weight policy training (no `[trainer.model.lora]`)
  and `enable_lora` on inference.
- Checkpoint GC: the orchestrator deletes dropped rollouts' adapter dirs at ship time and
  shipped ones once the weight watcher sees the trainer consume their step.

## Failure semantics

A failed update (service error, out-of-order version, missing token ids, failed Q&A
generation) is a `TTTError` — the rollout fails loudly (non-retryable 400 to the harness):
after a lost update the adapter no longer matches the context the model believes it has.
