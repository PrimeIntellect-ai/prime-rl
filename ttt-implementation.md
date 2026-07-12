# TTT implementation and decision record

This document describes the implemented compaction-time training path. `ttt-plan.md` is the
historical proposal; `docs/ttt.md` is the shorter operator-facing guide. The sliding-window
idea remains unimplemented.

The headline experiment is RL on ScaleSWE with GLM-4.5-Air. The smaller PEFT service and
DeepDive configs remain useful for local/eval experiments.

## 1. Core invariant

A TTT update happens at a compaction boundary, and every sampled branch is associated with
one adapter version.

Verifiers v1 represents a rollout as a message graph. The compacting harness replaces the
active conversation with `[system, user(summary)]`, which creates a new root-to-leaf branch
while preserving the abandoned full-context branch. The TTT hook observes that the prepared
turn no longer extends the previous leaf, trains on the abandoned branch, and stamps later
sampled nodes with the resulting adapter version.

This design deliberately does not support arbitrary mid-history edits. The supplied harness
creates a fresh summary branch, and `Branch.ttt_version` verifies that all sampled nodes on a
branch agree. Additional rewrite-history rejection and mixed-version sample splitting were
considered during review and removed: they duplicated verifier invariants, increased Echo and
trajectory complexity, and were not needed by the implemented compaction path.

Updates block the rollout. That is the simplest way to ensure the next inference request uses
the acknowledged adapter. A version-specific cache salt prevents prefix KV reuse after an
in-place adapter reload.

## 2. Compacting harness

The `compacting` harness is the default v1 program plus three knobs:

- `compact_at_tokens` (required and positive);
- `checkpoint_prompt`;
- `compaction_framing`.

After an assistant turn and its tool results are appended, the program estimates the next
prompt size. It uses provider prompt/completion usage when present and a conservative wire-size
estimate when usage is missing. On crossing the threshold, it requests a handoff summary with
the full conversation and the same tools still present, with `tool_choice="none"`.

The original message list is mutated only after a non-empty summary returns. Provider errors or
empty summaries therefore preserve the existing context rather than destructively replacing it
with an unusable branch.

Baselines follow directly: the plain harness has no compaction; `compacting` without TTT is the
summary-only baseline; `ttt.enabled = false` checks wiring without installing the hook.

## 3. Verifier rollout hook

`TTTRolloutHook` is attached to a rollout session when its env has active TTT config.

1. Before a model request, `on_turn_prepared` checks whether the new prepared path extends the
   previous leaf. A non-extension identifies the abandoned branch and triggers one update.
2. The payload uses the exact recorded token IDs. `loss_scope="all"` trains every newly
   abandoned token; `"sampled"` trains model-sampled tokens only. Shared and already-seen nodes
   are context.
3. The hook sends monotonic `seq_no` values and waits for the matching response.
4. On success it selects the rollout adapter, changes the cache salt, and stamps new trace nodes
   with the current version.
5. It records checkpoint paths, update loss/timing/token counts, and optional Q&A pairs in
   `trace.info["ttt"]`.
6. `aclose` releases the service state and unloads the adapter. `train_final_branch` is optional
   and off by default because its new version would not sample another turn.

Transient update/release/Q&A calls use bounded retries and resend the same payload. FSDP v2
enforces that assumption by comparing the full request fingerprint with the cached successful
request. PEFT v1 caches by sequence number and trusts the verifier to retry an identical
payload; it does not independently prove equality. Streaming is unsupported because the
interception path must commit and version a complete turn.

## 4. TTT service

The service exposes:

- `POST /update` with rollout identity, exact token IDs/mask, sequence number, and optional
  Q&A data;
- `POST /release`;
- `GET /health`.

Each successful update writes a standard PEFT checkpoint at
`<output_dir>/ttt/<rollout_id>/v<seq_no>/` and loads it into the configured vLLM replicas.

### 4.1 PEFT engine

The default engine loads a Hugging Face model with PEFT in one process. It keeps each live
rollout's adapter tensors and optimizer state on CPU, swaps one state into the shared wrapper,
runs a serialized update, and swaps it back out. It is intended for small models, tests, and
standalone evaluation.

The PEFT engine must not run under multi-process `torchrun`: every process would try to own the
same HTTP port and independent mutable model. Managed SLURM launch therefore requires the FSDP
engine, while PEFT remains an explicitly external one-process service.

PEFT v1 is a trusted-client compatibility path. It relies on verifier-generated safe rollout
IDs and matching adapter names, caches retries by sequence number rather than full payload,
and loads/unloads inference endpoints sequentially with best-effort release. Use FSDP v2 when
identity enforcement, exact retry conflicts, cancellation leases, or transactional replica
activation are required.

Its Q&A path uses the service's configured tokenizer name, remote-code setting, and optional
chat-template override. Inference admin calls use `admin_timeout_seconds`, as in FSDP v2.

### 4.2 FSDP/MultiLoRA engine

The FSDP engine reuses Prime-RL's custom trainer model stack and preallocates `max_slots`
MultiLoRA slots. The lowest free slot is claimed deterministically for a new rollout and reset
to a base-identical adapter. Each slot owns its optimizer; CPU optimizer offload is honored.

Rank 0 collects jobs for `max_batch_wait_seconds`, validates and materializes Q&A sequences
once, then broadcasts those exact jobs. Whole jobs are first-fit packed below
`max_tokens_per_forward`; tokens are ordered by slot so the segmented MultiLoRA routing is
correct. Each job keeps its own loss denominator and optimizer step even when the forward is
shared.

The distributed safety boundary is explicit:

- malformed/out-of-order jobs are rejected before slot mutation;
- capacity includes earlier new jobs in the same work order;
- only deterministic `ValueError` validation is isolated per job;
- an unexpected error after preparation is fatal to the worker group, avoiding peers entering
  different collective sequences;
- the control plane accepts unary `("stop",)` orders and broadcasts HTTP startup/exit failures
  so non-master ranks are not stranded.

Gradient clipping uses TorchTitan's DTensor/EP-aware helper. The engine checks finite losses,
gradients, parameters, and optimizer state on all ranks. Checkpoint tensors are also checked
before the rank-0 atomic directory rename.

Packed TTT currently requires flash attention, a chunked fused LM head, text-only inputs, and
`cp = 1`. Grouped-expert MultiLoRA wrappers choose one adapter rather than segmented slots, so
grouped experts are rejected. Output-head adapters are rejected because the frozen fused head
cannot apply them. A TTT-local fused-head backward computes only the hidden-state gradient;
this avoids changing Prime-RL's shared LM-head implementation while eliminating an otherwise
large unused frozen-weight gradient.

### 4.3 Identity, retries, and replica activation

The managed FSDP service derives the only accepted adapter name from its configured prefix and
the rollout ID. Rollout IDs are safe single path components, so checkpoint creation and cleanup
stay below the service root. An adapter cannot be rebound to another rollout.

The last successful update stores its complete semantic fingerprint. Repeating that exact
sequence number returns the cached result; changing tokens, masks, Q&A, tools, or identity at
the same sequence number is a conflict.

FSDP vLLM activation fans out concurrently. A version is marked loaded only after every
replica succeeds. A partial or ambiguous failure unloads the adapter name everywhere before the
rollout lease is released. Release has bounded tombstones, is idempotent, and retries unload
until every replica confirms success or a structured "already absent" response. Request
cancellation and HTTP timeouts transfer the per-rollout lease to a background waiter, so retry
or release cannot overtake work that is still running.

## 5. Managed deployment

The multi-node layout is `inference | trainer | ttt`. A top-level `[ttt]` config together with
`deployment.num_ttt_nodes > 0` causes the launcher to:

- start one FSDP service under `torchrun` on the TTT nodes;
- fill the shared output directory, model, tokenizer, matmul precision, and every vLLM admin
  root;
- replace env TTT URLs equal to `"auto"` with the service URL while preserving explicit
  external URLs;
- wait up to `startup_timeout_seconds` for `/health` before scheduling rollouts;
- tear the service down with the rest of the job.

Managed validation is intentionally narrow. It checks the model/output/tokenizer contract,
adapter prefix, LoRA rank and slot capacity, retained train checkpoints, a launched inference
deployment, FSDP engine selection, finite collector delay, and fullgraph incompatibility. It
does not ban arbitrary vLLM overrides, force every model debug setting to match, introduce
global launch reservations, or change Prime-RL inference defaults.

The authoritative service model is the resolved trainer policy model, including configs that
set component-level model names without a shared top-level `[model]`. Only active envs whose
TTT URL is `"auto"` share the managed service's adapter prefix; explicit URLs remain external.

Each ScaleSWE arm has a distinct output path, job name, and sandbox label. This prevents the
supplied A0-A5 configs from colliding with one another, but it does not reserve a namespace
against two simultaneous launches of the same arm.

## 6. Q&A and policy recycling

When `ttt.qa` is set, the hook generates structured, self-contained question-answer items with
the abandoned branch still in context. Multiple seeded calls cover facts, successful and failed
approaches, hypotheses, setup, and tool behavior. Optional verification removes items the model
identifies as unsupported or context-dependent; malformed verification fails open.

Each Q&A exchange is committed as a `ttt_qa` side branch under the pre-update version. These
tokens have a separate budget and can receive RL credit without changing the main rollout's
turn/token limits. The adapter trains surviving pairs standalone, with system and tool text as
masked context and loss on answer tokens. `qa.also_train_rollout` includes the raw branch too.

Two CE paths update the policy's main weights:

- A4 `recycle_to_policy` renders each rollout's own pairs.
- A5 `meta_lessons` compares a completed group's pairs and rewards, then renders general
  lessons. Lessons remain group-owned until the first post-filter survivor ships; this prevents
  loss when the first member is filtered or the group spans trainer batches.

The two flags are mutually exclusive in the experiment configs and validation.

## 7. RL replay and checkpoint cleanup

`trace_to_samples` maps a branch's `ttt_version` to its recorded checkpoint and stores it in
`TrainingSample.ttt_adapter_path`. Prime-RL main's normal trajectory conversion already assigns
a shared sampled node to one flattened branch sample; TTT relies on that rule rather than
adding a second fork/dedup implementation.

The packer keeps one adapter path per microbatch. `TTTReplayManager` validates the PEFT
checkpoint, resolves supported linear modules, and adds the frozen LoRA delta in a forward hook.
The delta is cast to the module's observed output dtype rather than the storage dtype of a
master parameter. Hooks are armed before compilation; fullgraph compile is rejected because
per-microbatch activation intentionally changes eager hook state.

`TTTCheckpointGC` is conservative and in-memory:

- register a batch before sending it to the trainer, so a fast weight publication cannot
  overtake ownership bookkeeping;
- defer directories used by shipped samples until the matching policy version appears;
- delete directories belonging to conclusively errored/filtered rollouts;
- carry undecided overflow rollouts until a later batch;
- delete eval checkpoints immediately.

There is no persistent lifecycle manifest or trainer consumption marker. Crashes can leak
checkpoint files, and the final trainer batch can leave files when no later weight version is
published. This is storage leakage, not premature deletion or replay corruption; clean the run
directory after completion.

TTT token-batched environments count the actual serialized `TrainingSample` payload, including
auxiliary CE samples. Ordinary Prime-RL environments retain their existing trace-token batching
semantics. The general batching-policy question is documented separately on the local branch
`sebastian/deferred-ttt-review-issues-2026-07-12`.

## 8. Experiment matrix

The ScaleSWE overlays use a pinned dataset revision, disable live image filtering, and select
the Prime registry. TTT arms run a rank-16 attention-only FSDP service on one 8-GPU node.

| Arm | Treatment | Main comparison |
| --- | --- | --- |
| A0 | Pure RL, no compaction | Full-context anchor |
| A1 | Compaction only | A0: summary effect |
| A2 | Compaction + raw-branch TTT | A1: weights-as-memory |
| A3 | Q&A-only TTT signal | A2: extracted memory signal |
| A4 | A3 + per-rollout CE recycling | A3 and A5 |
| A5 | A3 + reward-conditioned group lessons | A3 and A4 |

TTT evals require the renderer client; A0/A1 keep Prime-RL's normal eval transport. This is an
experimental limitation, not silently "matched" by a new core config flag. A4/A5 also add
different model calls and CE token counts, so they are not compute-matched automatically.

## 9. Review scope decisions

The detailed review initially mixed TTT fixes with general Prime-RL hardening. The latter were
removed from this branch and written as separate issue notes on the local branch
`sebastian/deferred-ttt-review-issues-2026-07-12`, under
`docs/issues/deferred-from-ttt-review/`.

Retained fixes are local to TTT or necessary for its supplied runs: compaction preservation and
token estimation; managed FSDP launch/health; FSDP exact retry and identity checks; distributed job
preparation/fail-fast behavior; replica transactions; TTT payload batching; A5 sample ownership;
replay dtype; and the TTT-local frozen-head optimization.

Not retained here: global SLURM reservations, broad inference/model parity bans, persistent
checkpoint manifests and trainer markers, general TrainSink ownership/liveness rewrites,
generic eval renderer selection, core vLLM `load_inplace` cleanup, model-specific Nemotron
mapping, or changes to Prime-RL's shared fused LM-head backward.

## 10. Remaining validation

Unit tests cover config overlays, replay, sink behavior, PEFT HTTP routing, FSDP registry and
CPU update logic, FSDP retry identity, transactional replica helpers, cancellation leases, and
compaction behavior. The root TTT suite passes in an isolated CPU environment, as do focused
verifier and ScaleSWE taskset tests.

Still required before trusting an experiment result:

1. a Linux multi-rank FSDP smoke with real CUDA/DTensor clipping and checkpoint export;
2. a live multi-replica vLLM reload/release cycle, including a partial failure;
3. a shared-filesystem end-to-end train batch and frozen replay;
4. a short ScaleSWE rollout that compacts, updates, resumes through the adapter, and releases;
5. monitoring disk use because cleanup state is not crash-persistent.

The service base model is fixed at launch while the RL policy evolves. Replay applies the
recorded adapter delta to the current policy, so this is an experimental approximation rather
than an exact reconstruction of the sampler's historical base weights.

## Appendix: sliding-window TTT (not implemented)

The historical extension would drop old turn groups at a token threshold and train on each
newly abandoned segment. Message-level drops would preserve valid chat-template structure.
Mid-branch updates without a branch transition would violate the current one-version-per-branch
invariant and require a different sample/replay abstraction. No code or experiment config on
this branch implements that mode.
