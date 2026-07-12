# TTT detailed review and remediation report

- Review date: 2026-07-12
- Branch: `sebastian/ttt-2026-07-06`
- Integrated upstream: `origin/main` at `321c8aa82`
- Merge commit: `f66775182`
- Scope: Prime-RL TTT feature, modified verifier and research-environment submodules,
  ScaleSWE configs, deployment templates, tests, and documentation

## Executive result

The branch now contains the TTT feature plus TTT-specific correctness fixes. The earlier
review patch also changed broad Prime-RL behavior; those changes were removed. Their reasoning
is preserved as separate issue notes on the local branch
`sebastian/deferred-ttt-review-issues-2026-07-12`, under
`docs/issues/deferred-from-ttt-review/`.

The working tree is source/config/unit-test coherent and suitable for a controlled Linux GPU
smoke test. It is not production-proven: this macOS host cannot run the actual FSDP/DTensor
collectives, CUDA kernels, multi-replica vLLM servers, or shared cluster filesystem.

The old `ttt-bugreport.md` and the now-obsolete branch-level shared-node bug note are absent.
Prime-RL main independently landed shared sampled-node deduplication and the final NCCL
dispatch-gate fix, so this branch uses those upstream implementations.

No branch changes were pushed. The issue notes are committed separately from TTT and do not
modify the `main` branch.

## What the feature adds

| Area | Added behavior |
| --- | --- |
| Verifier harness | Automatic handoff-summary compaction with preserved trace branches |
| Verifier TTT hook | Compaction-boundary update trigger, exact token/mask payloads, adapter version stamps, retry handling, optional Q&A |
| PEFT service | Single-process small-model LoRA updates and PEFT checkpoint export |
| FSDP service | Resident MultiLoRA slots, cross-rollout packed updates, per-slot optimizers, distributed coordination, checkpoint export |
| Inference integration | Per-rollout versioned adapter load/reload/unload and cache salting |
| Orchestrator | Version-to-checkpoint sample mapping, QA recycling, group meta-lessons, TTT checkpoint GC |
| Trainer | Frozen LoRA replay while only policy base weights receive gradients |
| Deployment | Launcher-managed FSDP service role and readiness gate |
| Experiments | ScaleSWE A0-A5 overlays and small standalone DeepDive configs |

## How the implemented path works

1. The compacting harness estimates the next prompt size. At the threshold it asks for a
   non-empty summary while the full conversation is still present, then rebuilds the active
   messages as a fresh system-plus-summary branch. Failed summary generation leaves the old
   context untouched.
2. Before the next model call, the verifier hook sees that the prepared path no longer extends
   the previous leaf. It sends the abandoned path's exact tokens and loss mask to the TTT
   service and blocks for the result.
3. The service updates that rollout's LoRA, writes a versioned PEFT checkpoint, and loads the
   adapter into inference. Subsequent turns use the adapter name and a versioned cache salt.
4. Sampled nodes are stamped with the adapter version. Branch conversion resolves the stamp to
   the recorded checkpoint and puts it on `TrainingSample.ttt_adapter_path`.
5. The trainer packer keeps one checkpoint path per microbatch. Frozen replay hooks apply that
   LoRA during the forward; gradients pass through the delta into base policy weights.
6. The orchestrator defers used checkpoint directories until a corresponding policy version is
   observed, deletes conclusively dead/eval directories, and carries undecided overflow.

The supported harness creates a fresh compaction branch. It does not need a generalized
rewrite-history validator or mixed-version sample splitting. `Branch.ttt_version` remains the
single verifier-side invariant: sampled nodes in one branch must agree on the adapter version.

## Concrete TTT issues retained and fixed

### Compaction correctness

- Provider usage describes the old request, not necessarily the next prompt after assistant
  and tool output. The harness now includes new content and uses a conservative estimate when
  usage is missing.
- Empty or failed summary generation previously risked destructive message mutation. The
  original messages now change only after a valid summary returns.

### Managed launch blockers

- Launcher-managed PEFT would start one mutable single-process server per GPU on the same port.
  Managed mode now requires FSDP; PEFT must be launched externally as one process.
- A managed service with no launched inference nodes passed validation even though the SLURM
  role and admin-root wiring only exist alongside inference. That dead configuration is now
  rejected.
- The orchestrator could race the large FSDP service's model startup. It now waits for `/health`
  with a configurable deadline.
- Model, Q&A tokenizer, output path, matmul precision, adapter prefix, retained checkpoints,
  LoRA rank, and resident slot capacity are wired or checked where the managed path genuinely
  depends on them.
- Model auto-wiring now uses the resolved trainer policy model even when no shared top-level
  `[model]` exists, and keeps the nested FSDP model config synchronized in memory.
- Managed-service prefix checks apply only to envs using `base_url = "auto"`; explicit external
  services retain independent prefixes, while a managed service with no `"auto"` consumer is
  rejected as dead configuration.
- Train replay with `compile.fullgraph=true` is rejected because per-microbatch eager hook
  activation is an intentional graph break.
- The FSDP collector delay must be finite; an infinite value otherwise overflows
  `threading.Event.wait` and kills the collector thread.

### FSDP update correctness

- Rank-local Q&A tokenization could produce different work or fail on only one rank. Rank 0 now
  materializes sequences before broadcasting the exact jobs.
- Broad exception isolation after GPU mutation could let ranks diverge. Only pre-mutation
  validation errors are per-job; unexpected execution faults fail the worker group.
- Deferred slot claims did not count earlier new jobs in the same batch. Capacity is now
  consumed in authoritative work-order order.
- Duplicate rollout IDs and adapter ownership collisions are rejected before ambiguous result
  mapping.
- Generic `torch.nn.utils.clip_grad_norm_` was not the project's DTensor/EP-aware path. The
  service now uses TorchTitan clipping with the engine's EP flag.
- CPU optimizer offload in the FSDP model config was ignored. Per-slot optimizers now honor it.
- Unary `("stop",)` work orders indexed a missing second tuple item; the control loop now accepts
  the documented shape. HTTP startup/exit failures are broadcast as abort orders.
- Losses, gradients, parameters, optimizer state, and exported tensors are checked for
  non-finite values before a result is published.

### Unsupported adapter targets and memory use

- Grouped-expert MultiLoRA wrappers select one active adapter and cannot route a packed
  multi-slot forward. Grouped experts are rejected for FSDP TTT.
- Output-head LoRA cannot pass through the frozen fused head and is rejected. A no-match target
  list now fails startup instead of silently training nothing.
- The shared fused LM-head backward always constructs a vocabulary-sized weight gradient even
  when the TTT base weight is frozen. The feature now installs a TTT-local hidden-gradient-only
  backward, avoiding a broad Prime-RL core edit.

### Retry, identity, replica, and cancellation safety

- A duplicate sequence number previously replayed a cached result without proving that the
  request was identical. FSDP slots now store and compare a fingerprint of every semantic field.
- Rollout IDs directly form checkpoint paths. Managed FSDP requests now require safe IDs and
  the exact adapter name derived from the service prefix; adapter ownership cannot be rebound.
- Releases need to be idempotent after a lost response. A bounded tombstone registry preserves
  identity and lets unload be retried after the slot is freed.
- Sequential replica loads could leave only part of inference updated. FSDP loads fan out,
  reconcile partial/ambiguous failures by unloading everywhere, and mark a version loaded only
  after all replicas succeed.
- A cancelled or timed-out HTTP request could otherwise release ownership while queued work
  continued. The per-rollout lease transfers to a background waiter until the work or
  release/unload transaction finishes.
- `server_v2.py` referred to two helper functions that did not exist after scope reduction,
  making the module fail at runtime. The response-validation and load-completion helpers are now
  local and covered by tests.

### PEFT configuration contracts

- PEFT Q&A loading ignored the service tokenizer name, remote-code option, and chat-template
  override. It now uses the configured tokenizer contract rather than the model default.
- PEFT inference admin calls hard-coded a 120-second timeout. They now honor
  `admin_timeout_seconds`.

### Sink, replay, and GC interactions

- A5 meta lessons were attached to the first survivor before post-batch filtering. If that
  member was filtered or separated by a batch cut, the lessons disappeared. They are now
  group-owned and delivered exactly once with the first post-filter survivor.
- Meta-provider failures were converted to an empty result before the sink could count them,
  making a broken A5 arm look healthy. Provider failures now reach the sink's enrichment
  boundary, where they are contained and recorded in `ttt/meta_groups_dropped`.
- Meta lessons selected conditioning only from a rollout with a non-empty system prompt, so a
  tool-only rollout silently lost its tool schema. Conditioning selection now accepts either a
  system prompt or tools.
- TTT token batching used trace totals even though replay and auxiliary CE samples can enlarge
  the actual trainer payload. TTT-enabled environments use serialized sample lengths;
  non-TTT environments keep Prime-RL's existing accounting.
- GC registration happened after `sender.send`, so a fast trainer weight publication could
  overtake the deferred-directory record. Registration now happens before the send.
- A carried path that later became conclusively dead was deleted from disk but retained in the
  in-memory carry set. Dead paths are now removed from both.
- Replay cast frozen adapters to a master parameter's storage dtype, which can differ from the
  actual BF16 forward dtype. Deltas now use the module output dtype.
- Frozen replay now refuses a checkpoint target that resolves to a non-`nn.Linear` module.

## Changes deliberately removed from the branch

The following were either general Prime-RL policy changes or defenses without a demonstrated
failure in the supplied TTT path:

- global SLURM submission reservations and unique submission identities;
- persistent checkpoint manifests, trainer-consumption markers, crash reconstruction, and
  fail-closed recovery state;
- general TrainSink rollout-window ownership, empty-cohort persistence, and liveness rewrites;
- global token-batch semantics and progress-token changes for ordinary environments;
- generalized verifier rewrite/fork rejection and mixed-version sample splitting;
- Echo objective changes needed only by that generalized splitting;
- arbitrary vLLM override bans and broad model/inference parity validation (RoPE, debug flags,
  dtypes, index cache, `trust_remote_code`, per-role overrides, FP32 head mode);
- generic eval renderer-client selection for non-TTT baselines;
- global inference LoRA-capacity validation;
- core vLLM `load_inplace` exception cleanup;
- shared fused LM-head backward changes;
- Nemotron-H adapter export/replay mapping unrelated to the GLM experiment.

General Prime-RL items from this list that have standalone value are captured as detailed issue
notes on the dedicated deferred-issues branch. TTT-only defenses are explained here instead.
Two general problems were independently fixed by current main while this work was underway:
shared sampled-node deduplication (`b453f40d8`) and the final NCCL dispatch gate
(`f34efdbf4`).

## Tests and verification

The final root TTT suite is intentionally consolidated around behavior rather than the earlier
large matrix of defensive assertions. It covers:

- all shipped service and ScaleSWE configs;
- Q&A rendering and replay/checkpoint mapping;
- PEFT HTTP update/release forwarding;
- PEFT tokenizer and admin-timeout configuration;
- FSDP slot allocation, packing, poisoned-job isolation, capacity, exact retry, release
  identity, frozen-head backward, and CPU update flow;
- transactional replica helper behavior, cancellation leases, identity rejection, unary stop,
  and HTTP startup failure;
- TrainSink TTT payload batching and A5 group ownership;
- meta-lesson failure accounting and system/tool conditioning;
- existing checkpoint GC behavior.

At the time of this report, the isolated CPU run passes 106 tests. Focused verifier tests pass
for the compacting changes, verifier lint passes, the ScaleSWE dataset-revision test passes,
the root lock validates, changed Python sources compile and lint, and rendered managed/baseline
SLURM scripts pass `bash -n`.

Independent review was attempted in three areas. The trajectory reviewer confirmed that the
verifier already enforces one version per branch and recommended removing generalized rewrite
splitting. The runtime reviewer found the broken `server_v2` imports. The workspace then ran
out of subagent credits, so those agents could not complete a final post-fix reread; the main
agent compensated with a clean isolated environment, combined-suite execution, and direct
source/diff review. This is a tooling limitation, not a claim of independent sign-off.

## Remaining limitations and unclear boundaries

### Real distributed execution is untested

There has been no real multi-rank FSDP update, CUDA fused-head backward, DTensor optimizer
offload, NCCL/Gloo failure, or live vLLM adapter transaction on this host. A Linux GPU smoke is
still a merge gate for confidence in the FSDP path.

### PEFT is a simpler standalone path

The ScaleSWE deployment uses FSDP. The PEFT server keeps its original simpler sequential admin
and best-effort unload behavior; the new identity, lease, tombstone, and replica-transaction
machinery is in FSDP v2. Treat PEFT as a small single-process/single-inference-endpoint path
with a trusted verifier client: it caches retries by sequence number, relies on verifier-safe
rollout/adapter identity, and does not validate token IDs against the model vocabulary before
GPU execution. Use it only within those limits unless it receives its own focused hardening
review.

### Checkpoint GC is not crash-persistent

GC state exists only in the orchestrator process. A crash can leak files, and the final batch
can retain its checkpoints when no later policy version is published. The removed persistent
manifest would address that but was judged too large and invasive for this branch. Current
behavior favors leaking storage over early deletion.

### The TTT base model becomes stale during RL

The service loads the policy base at launch while the trainer continues updating its weights.
Frozen replay applies the recorded adapter delta to the current policy, not the exact historical
base. This is part of the experiment's approximation and should be measured, not described as
exact sampler reconstruction.

### Baseline eval transport is not matched

TTT evals require renderer token IDs; A0/A1 retain normal Prime-RL eval routing. A generic core
flag was deliberately removed. Comparisons should disclose this possible confound.

### Compute is not matched

A3-A5 add Q&A calls; A4/A5 also add CE tokens. Wall time, tokens, and FLOPs differ across arms.
Telemetry can measure delivery, but the configs do not automatically normalize compute.

### Operational trust boundary

The service has no authentication or TLS and is intended for the job's private network. FSDP
validates identity and paths, but it is not an internet-facing multi-tenant service.

## Recommended smoke sequence

1. Launch FSDP TTT on one Linux GPU with a tiny supported model; perform update, retry, load,
   release, and replay.
2. Repeat with two ranks and verify packed updates, clipping, finite checks, and checkpoint
   export.
3. Use two vLLM replicas and inject one load/unload failure to verify reconciliation.
4. Run a short compacting ScaleSWE rollout through an adapter update and subsequent sampled
   turn.
5. Ship one RL batch, verify frozen replay uses the recorded checkpoint, and inspect GC.
6. Monitor the final-step directory for the documented non-persistent cleanup leak.
