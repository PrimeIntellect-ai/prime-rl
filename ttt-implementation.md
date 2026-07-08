# TTT Implementation (prime-rl)

Companion to `ttt-plan.md`. **Status: implemented** — this document describes the system as
built, plus the decision log. Branches: `prime-rl` `sebastian/ttt-2026-07-06`,
`deps/verifiers` `sebastian/ttt-compact-harness-2026-07-06` (pinned by exact commit in the
superproject). User-facing docs: `docs/ttt.md`; experiment configs: `configs/ttt/`
(RL family: `configs/ttt/scaleswe/`); the general branch-sample double-counting bug found
along the way is written up separately in `branch-sample-dedup-bug.md` (for an upstream PR).

**Scope:** the headline technique is **training at compaction time** (with and without Q&A).
The sliding-window technique is tabled — its design is kept in Appendix A; nothing in the
compaction path forecloses it. Experiments run **RL-first on `scaleswe-v1`** (GLM-4.5-Air,
SWE-Bench Verified evals); the earlier eval-only deepdive arms remain available in
`configs/ttt/` as the small-scale path.

## 0. Core design decisions

### 0.1 The one invariant that makes everything compose

**TTT updates happen only at compaction boundaries, and every trace branch is sampled under
exactly one LoRA version.**

verifiers v1 already gives us the perfect substrate: a rollout is a message *graph*; a
compaction (the harness rebuilds its prompt as `[system, user(summary)]`) forks a new
**branch**, and a branch is already the unit of a training sample (`Branch.token_ids` /
`sampled_mask` / `logprobs`). Aligning "one TTT update" with "one branch fork" means:

- the TTT trigger is *passive*: the interception layer detects the fork (`prepare_turn`
  returns a prefix that doesn't extend the previous leaf) and runs one update on the
  just-abandoned branch before serving the first turn of the new branch;
- each committed node is stamped with the version its tokens ran under
  (`MessageNode.ttt_version`); `Branch.ttt_version` derives the branch's single version and
  *enforces* uniformity (raises on a mix) — exactly the metadata the RL replay needs;
- the harness stays a plain program (it just rewrites its prompt, as rlm does); it needs
  zero knowledge of TTT, vLLM admin, or token ids.

Compaction-only scope keeps the trigger simple: every fork is a compaction, the abandoned
branch is fresh content (the new branch shares at most the system-prompt prefix; shared
prefixes and already-trained nodes ride as loss-masked context via the hook's `seen` set),
and the branch ends with the summary turn generated with the full context in prompt —
exactly the ordering "training at compaction time" requires. An empty-tail guard keeps SDK
retries of committed turns from triggering spurious updates.

### 0.2 One LoRA per rollout, updated in place, checkpointed per version

- Adapter name: `ttt-{trace.id}` (unique per rollout), served by vLLM alongside the base
  model.
- LoRA starts zero-init (B=0) ⇒ identical to base model ⇒ created lazily on the *first*
  compaction; the first branch runs under the base model name with exact consistency.
- Every update: gradient step(s) → save PEFT-format checkpoint to
  `outputs/ttt/<trace_id>/v<k>/` (shared FS) → `POST /load_lora_adapter` (prime-rl's
  wrapper does in-place same-name reload) → subsequent generate calls use the adapter name
  **with a per-version cache salt** (`{adapter}-v{k}` merged into any existing salt) — the
  in-place reload keeps the lora id, so unsalted prefix KV from old weights would be
  silently reused. (Fixing this exposed a latent bug: the v1 `TrainClient` passed
  `cache_salt` inside vLLM `sampling_params` where it isn't a field; it now rides the
  top-level generate-request field, which also makes prime-rl's existing per-policy-version
  salting reach the engine.)
- Checkpoints persist after the rollout ends (adapter unloaded from vLLM) — the replay
  artifacts for RL; the orchestrator GCs them once consumed (§6).
- Failure semantics: any failed update, QA generation, or version mismatch is a `TTTError`
  (non-retryable 400 to the harness) — the rollout fails loudly; after a lost update the
  adapter no longer matches the context the model believes it has.

### 0.3 Where each piece lives

| Piece | Location | State |
|---|---|---|
| `compacting` harness (default harness + rlm-style compaction) | verifiers: `v1/harnesses/compacting/` (+ hooks in `harnesses/default/`) | done |
| TTT trigger hook, version stamping, cache salting, QA generation | verifiers: `v1/ttt.py`, driven by `v1/interception/server.py` + `v1/rollout.py` | done |
| `TTTConfig` / `QAConfig` (env-level, `--ttt.*`) | verifiers: `v1/ttt.py`, field on `EnvConfig` | done |
| TTT service — peft engine (v1, small models) | prime-rl: `src/prime_rl/ttt/{trainer,server}.py` | done |
| TTT service — fsdp engine (v2, large models/throughput) | prime-rl: `src/prime_rl/ttt/{trainer_v2,server_v2}.py` | done |
| Service config (`TTTServiceConfig`, engine union) + `ttt` entrypoint | prime-rl: `configs/ttt.py`, `entrypoints/ttt.py` | done |
| Branch → sample adapter refs on the wire | prime-rl: `orchestrator/trajectories.py`, `transport/types.py` | done |
| Frozen-adapter replay in the trainer | prime-rl: `trainer/ttt_replay.py`, wired in `trainer/rl/train.py` + packer | done |
| Q&A recycling / group meta-extraction into the policy | prime-rl: `orchestrator/trajectories.py` (`qa_recycle_samples`), `orchestrator/qa_meta.py`, wired in `train_sink.py` | done |
| Adapter checkpoint GC | prime-rl: `orchestrator/ttt_gc.py`, driven by the orchestrator | done |
| Launch validation (full-weight policy, `enable_lora`) | prime-rl-configs: `RLConfig.validate_ttt` | done |

## 1. The compacting harness (verifiers)

A **new built-in harness id `compacting`** — a subclass of `DefaultHarness`, not a change to
its behavior (the default harness gained only a subclass hook, `extra_program_args()`, and
its chat loop reads the full completion for `usage`). `CompactingHarnessConfig` extends the
default's knobs with:

- `compact_at_tokens: int` (required, positive — no baked-in number);
- `checkpoint_prompt` / `compaction_framing` overrides (None = rlm-mirroring built-ins).

The compaction logic lives in the shared program (mirrors rlm's `_compact_branch`): when a
turn's reported `usage.prompt_tokens` crosses the threshold (checked after tool results are
appended, at most once per loop turn), ask the model for a handoff summary **with the full
conversation still in context** — the summary turn is the last turn of the old branch —
advertising the same tools with `tool_choice="none"` (identical rendered system block, so
the trace branches exactly at the rewrite), then rebuild messages as
`[system, user(framing + summary)]`.

Baselines fall out for free: plain harness = no compaction; `compacting` without `ttt` =
the compaction-only baseline; `ttt.enabled=false` = wiring ablation.

## 2. The TTT trigger hook (verifiers)

`TTTRolloutHook` (attached to the `RolloutSession` by the `Rollout` when `EnvConfig.ttt` is
set; driven by the interception server around each turn):

1. **Fork detection** (`on_turn_prepared`): the prepared turn doesn't extend the previous
   leaf → the branch was abandoned → one blocking update on it. Empty-tail guard for SDK
   retries.
2. **Payload**: the branch's flat exact token ids + a loss mask — `loss_scope="all"`
   (default: memory formation; tool outputs count) or `"sampled"` (ablation); shared-prefix
   and already-trained nodes are context; strict `seq_no = version + 1` ack.
3. **Model/sampling switch**: after the ack, `turn_model()` returns the adapter name and
   `turn_sampling()` the per-version-salted sampling (`RolloutSession` consults both).
4. **Stamping** (`after_commit`): new nodes get `ttt_version` (and `ttt_qa` for QA
   exchanges).
5. **Tools capture** (`capture_request`): tool schemas are lifted from the harness's own
   request bodies — used for QA generation and shipped with updates (§5).
6. **Lifecycle**: optional `train_final_branch` at rollout end; `aclose()` always releases
   the adapter (service slot + engine), checkpoints stay on disk.
7. Metrics per update (loss, token counts, seconds, QA stats) → `trace.info["ttt"]`.

TTT requires the renderer (train) client — updates consume exact token ids; the eval relay
fails loudly. Streaming turns under TTT are refused.

## 3. The TTT service (prime-rl, fourth process type)

`uv run ttt @ config.toml` (`TTTServiceConfig`). HTTP surface (both engines):
`POST /update {rollout_id, adapter_name, token_ids, loss_mask, seq_no, qa_pairs,
train_rollout, system_prompt, tools}` → gradient step(s) + versioned PEFT checkpoint +
in-place `/load_lora_adapter` on every `inference_admin_urls` entry; `POST /release`;
`GET /health`. No tokenizer on the raw-branch path (exact ids in); Q&A pairs are the one
text input (§5).

Two engines (`[engine] type`, discriminated union):

- **`peft`** (default; `trainer.py`/`server.py`): single-device HF + PEFT, one resident
  adapter, per-rollout CPU tensor + optimizer state swapped in/out per update. Small models
  (≤~8B), CPU tests. Checkpoints written in the PEFT format vLLM consumes.
- **`fsdp`** (v2; `trainer_v2.py`/`server_v2.py`): the prime-rl trainer stack —
  `setup_model` (custom modeling / FSDP2 / CP / AC / fused LM head, configured via
  `engine.model` overrides for numerics parity with the RL trainer) with
  `engine.max_slots` **resident MultiLoRA adapter slots** (the multi-tenant LoRA machinery:
  `apply_lora_to_model`, slot-registry `MultiRunManager`, per-slot AdamW over
  `get_named_parameters_for_run`, `reset_parameters(idx)` = B=0 zero-init on claim,
  deterministic lowest-free-index). **Cross-rollout batched updates**: whole jobs pack into
  forwards under `max_tokens_per_forward`; slot-ordered tokens ride the segmented
  `lora_num_tokens` layout; each job's loss is normalized by its own token count; per-slot
  optimizers step independently — throughput scales with tokens, not update count. Rank 0
  serves HTTP + coalesces jobs (`max_batch_wait_seconds`), validates per-job *before*
  broadcasting (a malformed job 409s alone), broadcasts work orders to all ranks. Slot-
  sliced adapter export via `get_state_dict_for_run` + `convert_adapter_to_hf` +
  `save_lora_config` — byte-compatible with the peft engine's checkpoint format. Launch
  under torchrun on the TTT node(s).

vLLM constraint for MoE deployments: LoRA serves attention projections only — keep
`lora.target_modules` attention-only (`q/k/v/o_proj`) for e.g. GLM-4.5-Air.

## 4. Deployment wiring

- Inference: `enable_lora = true`, `max_lora_rank >= lora.rank`,
  `max_loras >=` concurrent in-flight TTT rollouts (co-size with
  `orchestrator.max_inflight_rollouts` and the fsdp engine's `max_slots`).
- Orchestrator: `EnvConfig.ttt` flows through to the spawned env server untouched (the
  orchestrator EnvConfig inherits `vf.EnvServerConfig`); the eval CLI gets `--ttt.*` for
  free.
- Launch validation (`RLConfig.validate_ttt`): TTT envs require **full-weight policy
  training** (frozen TTT adapters don't stack on a trainable policy LoRA) and
  `enable_lora` on inference.
- NCCL weight broadcast is unaffected: adapters travel via filesystem + `/load_lora_adapter`.
  Service base-weight staleness under async RL is accepted (same class as `async_level`
  off-policyness; monitored via `mismatch_kl`) — replay exactness does not depend on it.

## 5. Q&A at compaction (v2 — model-authored, on-trace, verified)

When `ttt.qa` is set, each compaction runs **before** its update:

1. **Generation**: `qa.num_generations` parallel calls, each with the full abandoned branch
   in context plus one *seeded* instruction (`qa.seeds` — facts / what worked / what failed
   / theories / setup / tool behavior), under the current pre-update adapter, advertising
   the rollout's captured tools. The model authors **both questions and answers** —
   `qa.items_per_generation` structured `<item>` blocks per call (`type: qa | lesson`),
   extracted robustly (`parse_qa_items`) and near-dup-filtered (`dedup_items`). The prompt
   enforces the **self-containment contract** (pairs are trained context-free, so a
   question referencing "the conversation above" has no retrieval key) and phrases lessons
   with the **trigger condition as the question** ("When X and Y happens, what should you
   do?" → lesson + validity scope).
2. **On the trace**: each QA exchange is committed as a `ttt_qa`-tagged branch — real
   sampled tokens under a known adapter version, so **RL trains the memory-writing behavior
   itself** through the rollout's advantage. The tag excludes QA from `RolloutLimits` and
   the trace's turn/token metrics (own `qa.max_tokens` budget);
   `graph.leaves(include_qa=False)` backs the main-branch views.
3. **Verification** (`qa.verify`, opt-in): one extra call re-presents the branch + all
   numbered items; flagged items (answer unsupported / question not self-contained) are
   dropped and recorded in `trace.info["ttt"]["qa_rejected"]`. Fails **open** on a
   malformed verdict.
4. **Adapter training**: the service renders each surviving pair **standalone** —
   `[system, question, answer]` via the chat template with the rollout's `tools=`
   (loss-masked; tool lessons learn next to the tool descriptions), loss on the answer
   tokens only — Cartridges-style: the knowledge must come from the weights.
   `qa.also_train_rollout` adds the raw branch back into the same update.

Two RL-time paths recycle QA into the **policy's** main weights (both ce-routed, riding the
normal training batch, `rl_weights` zero — kept strictly disjoint in experiments):

- `qa.recycle_to_policy` — each rollout's own pairs, rendered with the policy tokenizer
  under the same system+tools conditioning (`qa_recycle_samples`). The naive arm.
- `qa.meta_lessons` — group-level meta-extraction (`orchestrator/qa_meta.py`): after a GRPO
  group finishes, one call sees every rollout's pairs **with its reward** and distills
  contrastive general lessons (what high-reward attempts did that low-reward ones didn't).
  De-myopifies per-rollout extraction; enrichment-only (needs ≥2 rollouts with pairs, fails
  open). No cross-task lesson buffer (deliberate: buffers are a complexity class prime-rl
  doesn't support yet).

## 6. RL integration (frozen-adapter replay)

1. **Wire**: `TrainingSample.ttt_adapter_path` (msgspec, omit-default — the plain GRPO wire
   is unchanged). `trace_to_samples` resolves each branch's stamped `ttt_version` to the
   service's checkpoint path via `trace.info["ttt"]`; a stamped branch with no recorded
   checkpoint is a hard error (never a silent base-model replay).
2. **Sample dedup**: a sampled node is trainable **exactly once** across the trace — the
   first branch containing it keeps its mask; later branches (QA forks, subagent-style
   forks) re-carry it as context. This fixes a general latent bug (N× gradient weight on
   shared sampled prefixes) documented in `branch-sample-dedup-bug.md` for upstreaming.
3. **Packer**: bins never mix adapter paths (`_MicroBatchBin.can_add` constraint) — one
   frozen adapter per micro batch; the bin's path rides on the `MicroBatch`.
4. **Trainer**: `TTTReplayManager` — forward hooks on the adapter's target Linears add
   `scale·(x @ Aᵀ @ Bᵀ)` from no-grad bf16 tensors (LRU-cached per checkpoint). No
   parameters, no wrapping — composes with FSDP/AC/compile; gradients flow to base weights
   only, *through* the frozen adapter path; `activate(None)` restores the exact base
   forward. Activated per micro batch in the RL train loop; the importance ratio is honest
   because trainer logprobs are computed under the same weights the sampler used. Samples
   with no adapter ref (first branch) run adapter-free — also exact.
5. **GC** (`TTTCheckpointGC`): dropped rollouts' adapter dirs deleted at ship time; shipped
   ones once the weight watcher observes the trainer consume their step.
6. **Constraint**: full-weight policy training (validated at launch); router replay
   composes (routing decisions come from inference, which sampled *with* the adapter).

## 7. Experiments — RL-first on scaleswe (`configs/ttt/scaleswe/`)

GLM-4.5-Air on `scaleswe-v1`, compacting harness on prime sandboxes, eval on SWE-Bench
Verified under the identical inference regime as training — same compaction, same TTT
updates, same QA generation. The dispatcher routes TTT eval envs through the renderer
(train) client (TTT consumes exact token ids; the chat relay would refuse at the first
compaction), and eval adapters/checkpoints are dismissed as each rollout finishes (no RL
replay on eval). TTT arms:
`compact_at_tokens = 16384` at 64k total; fsdp-engine service on its own 8-GPU node
(torchrun); attention-only rank-16 adapters. Launch:
`uv run rl @ scaleswe/base.toml @ scaleswe/arm_<X>.toml` (+ the service for TTT arms).

| # | Arm | Config | Reads against | Isolates |
|---|-----|--------|---------------|----------|
| A0 | Pure RL, no compaction | plain harness, 64k | — | Full-context anchor |
| A1 | Compaction baseline | `compacting@16k`, no TTT | A0 | What summaries alone buy |
| A2 | Compaction + TTT | + `ttt` (raw branch) | **A1** | **Headline: weights-as-memory** |
| A3 | + QA | + `qa = {}` | A2 | QA as the adapter signal; RL'd memory-writing |
| A4 | + naive recycle | `qa.recycle_to_policy` | A3, vs A5 | Per-rollout pairs → policy (ce) |
| A5 | + meta-lessons | `qa.meta_lessons` | A3, vs A4 | Group-contrastive lessons → policy (ce) |

A4/A5 are strictly disjoint (effects must stay disentangled). Primary metric: reward /
SWE-Bench score; secondary: per-update TTT loss, QA quality over training (sampled from
`trace.info["ttt"]` — the "learning to write memories" evidence), `mismatch_kl` (replay
health canary), compactions per rollout, reward-vs-rollout-length breakdown.

The eval-only deepdive arms (full/small/compaction-only/±TTT/±QA) remain in `configs/ttt/`
as the small-scale path (peft engine); the eval CLI drives them via `--ttt.*`.

## 8. Status log

- **Phase 1–5 (original plan)**: done — compacting harness; hook + service (peft engine);
  Q&A; RL replay; configs/docs/skills.
- **QA v2** (review round): model-authored items, seeded generations, structural
  extraction, self-containment contract, `ttt_qa` trace branches, system+tools
  conditioning, `qa.verify`, `qa.meta_lessons`, sample dedup fix.
- **fsdp engine (v2)** + `configs/ttt/scaleswe/` A0–A5 family: done.
- Tests: verifiers `tests/v1/test_ttt.py` + `tests/test_compacting_harness.py`; prime-rl
  `tests/unit/ttt/` (trainer math on a tiny on-disk llama, QA rendering with an offline
  tokenizer, HTTP surfaces against fakes, replay-hook math vs a hand LoRA reference, slot
  registry/packing, GC, meta-extraction). All green; lint clean.
- **Remaining (needs the GPU box)**: `uv lock` regen (peft dep added; lock env is
  linux-only); first real run of the fsdp engine's collective path (smoke on 1 GPU with a
  tiny model first); verify vLLM `enable_lora` × TP=8 × `enable_return_routed_experts` on
  GLM-4.5-Air; deepdive/scaleswe smoke with the compacting harness + eyeball QA output
  (decides whether `qa.verify` defaults on); draft PRs; the separate upstream PR for the
  dedup bug.

## 9. Decision log

1. Loss scope — `all` tokens (context extension: every token counts); `sampled` as ablation.
2. Environment — RL-first on `scaleswe-v1` (was: eval-first on `deepdive_v1`, kept as the
   small-scale path).
3. Blocking updates — accepted (correctness over latency); the fsdp engine's batching
   recovers throughput across rollouts.
4. Sliding window — tabled (Appendix A).
5. Base-weight staleness under async RL — accepted, monitored via `mismatch_kl`.
6. Final-branch update — default off (`train_final_branch`).
7. Q&A budget — separate from `RolloutLimits` (`qa.max_tokens`); QA exchanges are `ttt_qa`
   trace branches: RL'd, but excluded from rollout metrics/budgets.
8. `max_loras` sizing — co-size with `max_inflight_rollouts` and `engine.max_slots`;
   documented + launch-validated.
9. Compacting harness is a **separate harness id**, not a default-harness knob.
10. QA: model authors questions *and* answers; lessons phrased trigger-as-question;
    self-containment contract; system prompt + tools as loss-masked training context.
11. During the rollout the adapter is **SFT'd** on standalone pairs; for RL the QA
    generations are trained **as branches** (what the model actually produced) — both, from
    the same pairs.
12. `recycle_to_policy` (naive) and `meta_lessons` (group-contrastive) are head-to-head
    alternatives, never combined; no cross-task lesson buffer.
13. Permanent-SFT comparisons must stay disentangled (A4 vs A5).
14. The branch-sample double-counting is a general bug, fixed here and documented for a
    standalone upstream PR (`branch-sample-dedup-bug.md`).

## Appendix A — sliding-window technique (tabled)

Kept for later; nothing in the compaction path forecloses it. Sketch: add
`context_mode="sliding_window"` + `window_tokens` to the compacting harness; after each
turn, drop oldest *turn groups* (assistant + its tool messages together; system + task
pinned) once the prompt exceeds the window → each drop forks a branch → the same TTT
trigger fires with a *watermark* per rollout (train only the not-yet-trained tail; the
retained window is context — the hook's `seen` set already implements most of this).
Message-granularity drops (not exact token slices) keep chat templates valid — the accepted
simplicity tradeoff. Mid-branch updates every N tokens (without dropping context) would
break the one-branch-one-adapter invariant and stay deliberately out of scope. The
experiment arms (32k full vs 8k window+TTT vs 8k window no-training vs 8k plain) are in
`ttt-plan.md`.
