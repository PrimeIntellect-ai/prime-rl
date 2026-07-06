# TTT Implementation Plan (prime-rl)

Companion to `ttt-plan.md`. Target repo: `prime-rl` (branch `sebastian/ttt-2026-07-06`), with a
paired branch on the `deps/verifiers` submodule (pinned by exact commit in the superproject).
Everything below reuses existing components wherever possible; all numeric values (thresholds,
LRs, ranks, steps) are config fields — nothing hard-coded.

**Scope decision (2026-07-06):** the headline technique is **training at compaction time** (with
and without Q&A). The sliding-window technique is tabled — its design is kept in Appendix A so it
can be picked up later, and nothing in the compaction path forecloses it. Environment:
`deepdive_v1`. Resolved defaults: TTT loss on **all** tail tokens (context extension = every
token counts); blocking updates accepted; base-weight staleness under async RL accepted with
`mismatch_kl` monitoring.

## 0. Core design decisions

### 0.1 The one invariant that makes everything compose

**TTT updates happen only at compaction boundaries, and every trace branch is sampled under
exactly one LoRA version.**

verifiers v1 already gives us the perfect substrate: a rollout is a message *graph*; a compaction
(the harness rebuilds its prompt as `[system, user(summary)]`) forks a new **branch**, and a
branch is already the unit of a training sample (`Branch.token_ids` / `sampled_mask` /
`logprobs`). Aligning "one TTT update" with "one branch fork" means:

- the TTT trigger is *passive*: the interception layer detects the fork (`prepare_turn` returns a
  prefix that doesn't extend the previous leaf) and runs one update on the just-abandoned branch
  before serving the first turn of the new branch;
- each branch carries a single `(adapter_name, adapter_version)` stamp — exactly the metadata the
  RL replay needs later;
- the harness stays a plain program (it just rewrites its prompt, as rlm already does); it needs
  zero knowledge of TTT, vLLM admin, or token ids.

Compaction-only scope makes the trigger especially simple: **every fork is a compaction**, and
the abandoned branch is entirely fresh content (the new branch shares at most the system-prompt
prefix). No partial-tail watermark bookkeeping is needed — the update trains on the whole
abandoned branch, which by construction ends with the summary turn generated with the full
context in prompt (exactly the ordering "training at compaction time" requires).

### 0.2 One LoRA per rollout, updated in place, checkpointed per version

- Adapter name: `ttt-{trace.id}` (unique per rollout), served by vLLM alongside the base model.
- LoRA starts zero-init (B=0) ⇒ identical to base model ⇒ created lazily on the *first*
  compaction; the first branch runs under the base model name with exact consistency.
- Every update: gradient step(s) → save PEFT-format checkpoint to
  `outputs/ttt/<trace_id>/v<k>/` (shared FS) → `POST /load_lora_adapter` (prime-rl's wrapper
  already supports in-place same-name reload) → subsequent generate calls use the adapter name.
- Checkpoints persist on disk after the rollout ends (adapter unloaded from vLLM) — they are the
  replay artifacts for the RL phase and for analysis.

### 0.3 Where each piece lives

| Piece | Repo / location | New or extended |
|---|---|---|
| Compacting default harness (rlm-style) | `deps/verifiers` branch: `verifiers/v1/harnesses/default/program.py` + `harness.py` config | extend |
| TTT trigger hook (fork detection → update call, adapter-version stamping, cache salting) | `deps/verifiers` branch: `verifiers/v1/interception/` + `clients/` | new, small |
| TTT config (`TTTConfig`) | `deps/verifiers` branch: `verifiers/v1/configs/` (flows through `EnvServerConfig`, and the `eval` CLI) | new |
| TTT update service (holds base model, computes LoRA steps, drives vLLM adapter loads) | `prime-rl`: `src/prime_rl/ttt/` + `ttt` entrypoint | new |
| vLLM adapter serving | `prime-rl`: existing `enable_lora` + `/load_lora_adapter` in-place reload | reuse |
| Branch → sample adapter refs on the wire | `prime-rl`: `orchestrator/trajectories.py`, `transport/types.py` | extend (RL phase) |
| Frozen-adapter replay in trainer | `prime-rl`: `trainer/rl/packer.py`, `trainer/rl/train.py`, MultiLoRA layers | extend (RL phase) |
| Q&A generation + recycling | verifiers hook (generation) + existing `ce_weights` loss routing (recycling) | new + reuse |

## 1. Component A — compacting default harness (verifiers branch)

Extend the **default harness** (uv-script chat loop, `harnesses/default/`) with rlm-style
compaction, per the decision to use "default harness with compaction like in rlm".

`DefaultHarnessConfig` additions (optional, default = today's append-only behavior):

```python
compact_at_tokens: int | None = None  # compact once prompt tokens cross this (rlm's SUMMARIZE_AT_TOKENS); None = never
```

Program changes (`program.py`, pure message-list logic → unit-testable as plain functions):

- Track the running prompt-token count from each response's `usage.prompt_tokens` (the same
  signal rlm uses; the renderer client reports exact counts).
- When the threshold is crossed, port rlm's `_compact_branch` semantics: append a checkpoint
  prompt and ask the model for a handoff summary **with the full conversation in context** —
  this summary turn is the last turn of the old branch and is therefore included in the TTT
  update. Advertise the same tools with `tool_choice="none"` on the summary turn (rlm's trick to
  keep the rendered system prompt identical so branching is clean and deliberate).
- Rebuild messages as `[system, user(framing + summary)]` and continue. The next request forks
  the trace — the compaction boundary the TTT hook keys on.
- Fires at most once per loop turn; threshold check after tool results are appended (mirrors rlm).

Baselines fall out for free:
- *full context* = `compact_at_tokens=None` (today's behavior).
- *small plain context* = `compact_at_tokens=None` + `RolloutLimits.max_input_tokens` (exists).
- *compaction without TTT* = compaction on, TTT off — same code path minus the update calls.

Note on `deepdive_v1`: it runs on the default harness with MCP search tools, so the compacting
default harness slots straight in; the env itself needs no changes.

## 2. Component B — TTT trigger hook (verifiers branch)

A small, framework-owned object attached to the `RolloutSession` when `TTTConfig` is present
(invisible to harnesses; works identically under the `eval` CLI and the env server — both drive
rollouts through the same interception path).

Behavior, in the interception request handler around `graph.prepare_turn(...)`:

1. **Fork detection**: keep the previous leaf's node path per session. If the new
   `PendingTurn`'s resolved prefix does not extend the previous leaf → the previous branch was
   abandoned → it was a compaction (the only rewrite the compacting harness performs).
2. **Update payload**: the abandoned branch as a flat sequence — `token_ids = branch.token_ids`,
   `loss_mask` = all tokens (`loss_scope="all"`, the resolved default: this is memory formation
   for context extension, so tool outputs and prompts count too; `"sampled"` kept as a config
   ablation). The shared system-prompt prefix is excluded from loss (it's in every branch).
3. **Call the TTT service** (`POST {ttt.base_url}/update`) with
   `{rollout_id, adapter_name, token_ids, loss_mask, seq_no}` and *block* until it acks
   (update applied + adapter (re)loaded into vLLM). Simplicity and exactness over latency
   (accepted: ~1–3 s per compaction at 8B/32k).
4. **Version bookkeeping**: bump the session's `ttt_version`; from now on the client's generate
   calls use `model = adapter_name` and a per-version **cache salt**
   (`extra_body.cache_salt = f"{existing_salt}-ttt{version}"`). The salt is mandatory: prime-rl's
   in-place adapter reload keeps the same `lora_int_id`, so vLLM's prefix cache would otherwise
   reuse KV computed under the *old* adapter weights (same issue prime-rl already solves for
   policy reloads with `cache_salt=policy_version`).
5. **Stamping**: record `ttt_version` on every `MessageNode` at commit time (new optional field);
   `Branch` gets a derived `ttt_version` property that asserts uniformity — the §0.1 invariant,
   enforced.
6. **Metrics**: per-update loss, token counts, wall time → `trace.info["ttt"]` (and surfaced as
   `@metric`s), so W&B/eval tables show TTT behavior per rollout.
7. **Teardown**: on rollout end (the `Rollout.run` finally), `POST /release` (unload adapter
   from vLLM, drop optimizer state; checkpoints stay on disk).

Error semantics: a failed update call is a rollout error (new `TTTError` boundary) — fail
loudly, no silent degradation.

`TTTConfig` (new, in `verifiers/v1/configs/`, reachable from `EnvServerConfig` /
`--ttt.*` on the eval CLI):

```python
class TTTConfig(BaseConfig):
    base_url: str                      # TTT service
    enabled: bool = True               # false = ablation wiring without updates
    loss_scope: Literal["all", "sampled"] = "all"
    train_final_branch: bool = False   # update on the last branch at rollout end (default off)
    qa: QAConfig | None = None         # §5
```

(LR/optimizer/rank live on the *service* config — the service owns training hyperparams; the
env-side config owns triggering. One place per knob.)

## 3. Component C — the TTT service (`src/prime_rl/ttt/`)

A fourth process type alongside inference/orchestrator/trainer, with its own entrypoint
(`ttt = "prime_rl.entrypoints.ttt:main"`) and `TTTServiceConfig` in `prime-rl-configs`.

- **Model**: loads the base model (HF `AutoModelForCausalLM`, bf16, gradient checkpointing) on
  its own GPU(s). For the experiment scale (≤8B dense) one GPU suffices; no FSDP in v1 of this.
- **Adapters**: PEFT `LoraConfig` per rollout (rank/alpha/dropout/target_modules from config).
  PEFT rather than prime-rl's MultiLoRA because (a) the service trains *one adapter at a time*
  per request, (b) PEFT's on-disk format is exactly what vLLM's `/load_lora_adapter` consumes.
  Per-rollout optimizer (configurable: `adamw` default, `sgd` option; LR, `steps_per_update`,
  grad clip all config).
- **API** (FastAPI, mirrors the inference server's style):
  - `POST /update {rollout_id, adapter_name, token_ids, loss_mask, seq_no}` →
    forward/backward/step(s) on the sequence (single-sequence batch), save checkpoint
    `outputs/ttt/<rollout_id>/v<k>/`, call vLLM `/load_lora_adapter` (admin URLs from config;
    reuse `prime_rl.utils.client.load_lora_adapter`), return `{version, loss}`. Requests for the
    same rollout are serialized; distinct rollouts run concurrently up to a semaphore
    (`max_concurrent_updates`).
  - `POST /release {rollout_id}` → unload adapter from vLLM, free optimizer state.
  - `GET /health`.
- **Weight updates under RL** (phase 2): the service follows the trainer's weight broadcasts
  exactly like vLLM does (watch `broadcasts/` dir, reload base weights per policy version) so
  TTT gradients are computed against the same base the policy serves. For the eval-only phase
  the base is static and this is inert. Staleness under async RL (adapter trained on base *v*,
  replayed against *v+lag*) is **accepted** — same class as existing `async_level`
  off-policyness, monitored via `mismatch_kl`.
- **Tokenization**: none. The service consumes exact token ids from the trace — no re-rendering,
  no drift.

Memory envelope: 8B bf16 ≈ 16 GB + LoRA/optimizer (MBs per rollout) + one ≤32k-token
activation-checkpointed fwd/bwd — fits one H100. Concurrency bound is compute, not memory.

## 4. Component D — vLLM/orchestrator wiring (prime-rl, minimal)

- Inference config: `enable_lora = true`, `max_lora_rank ≥ ttt rank`, `max_loras ≥` expected
  concurrent in-flight TTT rollouts (config validation cross-check + docs; adapters
  unloaded-mid-request is the crash edge the existing `max_cpu_loras` comment warns about —
  sizing rule documented, validation warning first).
- Orchestrator: pass-through of `TTTConfig` on `EnvConfig` (train + eval) into the spawned env
  server (it already forwards the full config), plus surfacing TTT metrics from `trace.info`.
- Nothing else changes in the orchestrator for the eval phase.

## 5. Component E — Q&A at compaction (Cartridges-style)

When `ttt.qa` is set, the trigger's compaction handling becomes:

1. The hook makes `qa.num_pairs`-worth of side generations *with the full abandoned branch in
   context* (prompted templates: knowledge contained, approaches that worked/failed, theories,
   task setup...). These go through the same intercepted client, so they are recorded on the
   trace as branches — tagged `ttt_qa` (new node/branch tag) so they are excluded from the main
   trajectory's samples and RL credit, but preserved as training data.
2. The update step then trains the LoRA on the Q&A dataset (`qa.train_lora: bool`, default true)
   instead of / in addition to the raw branch (`qa.also_train_rollout: bool`), matching the two
   arms in the plan's experiments.
3. Recycling into main weights (RL phase): Q&A branches ship as `TrainingSample`s routed to the
   **`ce` loss component** (the `ce_weights` stream + `action_loss_type` machinery already
   exists — echo uses it today). No new trainer loss code: the "one SFT step after RL" becomes
   "Q&A samples ride the same batch with ce routing".

`QAConfig`: `num_pairs`, prompt template(s), sampling overrides, `qa.max_tokens` (Q&A budget is
counted separately from `RolloutLimits` — housekeeping, like rlm's summary call), `train_lora`,
`also_train_rollout`, `recycle_to_policy` (phase 2).

## 6. Component F — RL integration (phase 2)

The full replay design, building on §0.1's one-adapter-per-branch invariant:

1. **Wire**: `TrainingSample` gains `ttt_adapter_path: str | None` (msgspec, omit-default so
   the plain GRPO wire is unchanged). `trace_to_samples` reads the branch's stamped version and
   resolves the checkpoint path (`outputs/ttt/<trace_id>/v<k>/`, shared FS between orchestrator
   and trainer — same assumption the weight broadcast dir already makes).
2. **Packer**: group samples by adapter; v0 simplification: **one adapter per microbatch**
   (compaction rollouts have few branches, each a long sample — packing loss is small). The
   segmented `lora_num_tokens` layout MultiLoRA already computes with allows mixing later.
3. **Trainer**: a "frozen context adapter" mode for the MultiLoRA layers — per microbatch, load
   the referenced checkpoint into an adapter slot (small H2D copies), `requires_grad=False`,
   set `lora_num_tokens`. Gradients flow to base weights only; the loss stack (importance ratio,
   advantages) is untouched — and the ratio is *honest* because the trainer's logprobs are
   computed under the same adapter the sampler used. Samples with `ttt_adapter_path=None`
   (first-branch, pre-first-compaction) run adapter-free — also exact.
4. **Constraint**: policy trained with full weights (not policy-LoRA) in TTT experiments —
   stacking a trainable policy LoRA on frozen TTT LoRAs is possible with MultiLoRA's slots but
   not worth the complexity now; enforce via config validator.
5. **Checkpoint GC**: orchestrator deletes a rollout's adapter checkpoints after its batch ships
   (train sink hook), TTL fallback for errored/filtered rollouts.
6. **Q&A recycling** per §5.3.

## 7. Experiments mapping (what config runs what)

Environment: **`deepdive_v1`** (multi-turn MCP search over a corpus; long, information-dense
rollouts; default harness; existing scoring). All runs use the **renderer (train) client even
for eval** — TTT needs exact token ids.

Compaction suite (all knobs in TOML; numbers illustrative):

| Arm | Harness | TTT |
|---|---|---|
| full context | `compact_at_tokens=None`, `max_input_tokens=32k` | off |
| small plain context | `compact_at_tokens=None`, `max_input_tokens=8k` | off |
| compaction only | `compact_at_tokens=8k` (total budget 32k) | off |
| compaction + TTT | same | on |
| compaction + Q&A-TTT (context-extension only) | same | on, `qa` set, `train_lora=true` |
| (RL phase) + Q&A→policy | same | on, `qa.recycle_to_policy=true` |

Deliverable: `configs/ttt/` (or `examples/ttt/`) with one TOML per arm + a README with launch
commands (eval first via the v1 `eval` CLI / orchestrator eval mode; RL later via `rl`).

## 8. Phasing & deliverables

- **Phase 1 — compacting default harness (no TTT at all).** verifiers branch: default-harness
  `compact_at_tokens` + unit tests on the pure compaction functions + branch-fork behavior.
  Unlocks the compaction-only baseline and de-risks branching/renderer-bridge behavior.
  Smoke-eval on `deepdive_v1` against a live model.
- **Phase 2 — TTT service + trigger hook (eval-only TTT).** `src/prime_rl/ttt/` service;
  interception hook + `TTTConfig`; vLLM adapter wiring + cache salting; node/branch version
  stamping; metrics. Unit tests: service step math (update changes outputs; checkpoint
  round-trips through vLLM's PEFT loader format), fork-detection trigger against a scripted fake
  client, cache-salt/version bookkeeping, branch-uniformity assertion. GPU integration test on a
  remote box (tiny model): rollout with 2 compactions → 2 adapter versions → trace stamps
  consistent, generations after update differ from before.
- **Phase 3 — Q&A.** Side-generation + tagging + LoRA-training arms; config-driven templates.
- **Phase 4 — RL replay.** Transport field, packer grouping, frozen-adapter trainer mode,
  GC, Q&A→ce recycling. Unit tests: packer grouping/segment layout; frozen-adapter forward
  equivalence (loading a known adapter reproduces reference logprobs); ce-routing of QA samples.
- **Phase 5 — experiment configs + docs** (`configs/ttt/`, skills update per repo policy).

Each phase = reviewable PR(s) on the two branches; verifiers submodule pinned by commit in
prime-rl at every step.

## 9. Resolved decisions (review log)

1. **Loss scope** — `all` tokens by default (context extension: every token in the window
   counts), `sampled` kept as ablation config. ✅ resolved 2026-07-06.
2. **Environment** — `deepdive_v1`. ✅
3. **Blocking updates** — accepted (correctness over latency). ✅
4. **Sliding window** — tabled; compaction experiments first (see Appendix A). ✅
5. **Base-weight staleness under async RL** — accepted, monitored via `mismatch_kl`. ✅
6. **Final-branch update** — default off (`train_final_branch=false`), config-flippable.
7. **Q&A token accounting** — separate `qa.max_tokens` budget, not counted against
   `RolloutLimits` (housekeeping, like rlm's summary turn).
8. **`max_loras` sizing** — document + config-validation warning first; hard dispatcher cap
   only if it bites in practice.

## Appendix A — sliding-window technique (tabled)

Kept for later; nothing in the compaction path forecloses it. Sketch: add
`context_mode="sliding_window"` + `window_tokens` to the default harness; after each turn, drop
oldest *turn groups* (assistant + its tool messages together; system + task pinned) once the
prompt exceeds the window → each drop forks a branch → the same TTT trigger fires with a
*watermark* per rollout (train only the not-yet-trained tail; the retained window is context).
Message-granularity drops (not exact 1k-token slices) keep chat templates valid — the accepted
simplicity tradeoff. Mid-branch updates every N tokens (without dropping context) would break
the one-branch-one-adapter invariant and are deliberately out of scope. The experiment arms
(32k full vs 8k window+TTT vs 8k window no-training vs 8k plain) are in `ttt-plan.md`.
