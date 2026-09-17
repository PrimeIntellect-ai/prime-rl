# Stage 1: proposed prime-rl implementation

## Existing flow and the missing abstraction

`TrainSource.next_task` → `TaskRequest` → `Dispatcher` → episodes/failures/cancellations → `TrainSink` → `Algorithm.finalize_group` → curriculum admission → tokenization/packing → trainer.

Relevant files:

- `src/prime_rl/orchestrator/train_source.py`: task selection, source mixing, curriculum callbacks, checkpointed sampler state. No continuation queue.
- `dispatcher.py`, `types.py`: async scheduling already supports requested rollout counts and group IDs. Every physical group has a single dispatch version and terminal accounting.
- `train_sink.py`: waits for configured group_size, scores immediately, queues traces individually, drops stale traces and slices batches by trace count.
- `algo/base.py`, `algo/grpo.py`: native-episode hooks and centered reward credit; no scheduling result or historical baseline context.
- `ckpt.py`: saves progress and TrainSource state, not unfinished sink cohorts or algorithm state.
- `packages/prime-rl-configs/src/prime_rl/configs/algorithm.py`: named typed algorithm variants.

A curriculum-only implementation is insufficient: `on_result` returns a boolean after scoring and cannot supply historical baseline statistics or postpone terminal scoring. A retry loop inside an env would also be wrong: it hides independently sampled completions, their behavior policies, and compute from the orchestrator.

## Proposed ownership

1. **Physical round**: retain the dispatcher's existing K-sized group, fresh group ID, current dispatch version, and failure/cancellation accounting.
2. **Logical visit**: introduce a chain ID that follows multiple rounds. Key state by source + visit ID; record task key/hash as provenance. Independent visits to the same task must not share history.
3. **Continuation queue**: TrainSource accepts a retry TaskRequest for the exact same task and chain, with a new physical group ID. Each finished failed round can enqueue at most one next round. Service queued retries FIFO while existing work proceeds; instrument fresh-versus-retry wait times. Source ratios describe fresh selections, not resulting compute shares.
4. **Group lifecycle**: a small controller at the sink boundary merges a completed round into visit state and returns a typed decision: retry, discard, or finalize with retained episodes and immutable history statistics. Standard GRPO's default is immediate finalize. Scheduling remains outside Algorithm.
5. **NGUAlgorithm**: named `ngu` implementation assigns anchored advantages to an eligible finalized cohort; reuse loss routing, trace conversion and trainer transport. Pass history as explicit group context, not as fake episodes or rewritten task rewards.

Initial proposed config contract (not implemented): `[orchestrator.algo] type="ngu"`, `continuation_probability`, `history_max_policy_age`, `seed`. Centering and positive anchoring are the named algorithm's defaults. `p=0` must reduce to standard GRPO credit and filtering on valid binary cohorts. Both arms use unshaped binary rewards. Require `0 <= p < 1` in production configs; p=1 requires a separately bounded diagnostic if ever needed.

## Correctness decisions before coding

- Initially support live-policy, single-agent, one-trainable-trace episodes with raw task rewards exactly 0 or 1. No length penalty or reward shaping; use binary solved reward for continuation, historical counts and credit. Reject incompatible algorithm/env combinations at config resolution when knowable; validate reward values at runtime.
- Count valid scored attempts only in C/S. Transport failures, verifier errors, cancellations and empty/untrainable results are not reward-zero samples. Still count their compute and terminal accounting. Proposed policy: terminate affected chains on incomplete/error rounds for an unambiguous first implementation; log this separately from probabilistic give-up.
- Expire payloads by the oldest actual generation version: `age = (training_step - 1) - policy.start`. Preserve C/S. The effective bound is the minimum of history_max_policy_age and the existing trainer staleness bound; do not change either policy provenance or numerical dtype settings.
- Re-score after freshness filtering at the point of shipment. If either reward class is absent, drop the cohort with an explicit reason. Never retain expired negatives just to balance positives.
- **Batch policy needs an explicit change**: current trace slicing can split a finalized cohort across optimizer updates. Prefer whole-cohort FIFO packing with a nominal trace target and final-cohort overshoot, across *all experiment arms*. Actual batch size must be measured. This changes exact fixed-trace batching, so keep it explicit and preserve existing default behavior for unrelated runs. If exact-size batches are required, use a separately reviewed balanced-subset policy; silently truncating is not acceptable.
- Even within an age window, fast retries can accumulate many payloads at one policy version. Add a visible retained-token/byte limit and bound active visits; payload eviction keeps historical counts and logs the event. No hidden retry cap masquerading as paper NGU.
- The sink's current 10 zero-output-batch guard can fire during intentional persistence. Keep a fail-fast progress guard, but distinguish queued productive work from a prolonged no-signal run; expose any experiment-specific threshold rather than disabling it globally.
- Separate accounting for generated episodes and later trained episodes. Reporting a buffered failure again when its chain succeeds must not double-count tokens, rewards, traces, or samples.
- New task requests after a retry must use the current step/version; never reuse the original physical group's stale dispatch version.

## Resume and cancellation

Persist continuation RNG, chain IDs/provenance, C/S, retained payload references, retry queue, and completed-round markers through a unified state owner. Snapshot completed rounds consistently with training progress. Do not pickle asyncio tasks. On resume, reschedule incomplete physical rounds with new IDs and discard their incomplete output; preserve only committed round statistics. Exactly-once completed-round ingestion prevents duplicate negative counts.

This is stronger than current orchestrator resume, which restores source state but not all pending payloads. If exact recovery is deferred, explicitly clear and count unfinished chains on restart, label the run non-exact, and exclude interrupted runs from the adoption experiment. Do not claim faithful NGU resume from sampler state alone.

## Implementation sequence and verification

1. Implement the isolated binary visit state machine and anchoring transformation, with explicit invariants.
2. Add typed config, named algorithm and lifecycle result; default lifecycle keeps static behavior.
3. Connect continuation requests through TrainSource/Dispatcher; leave physical-round completion accounting intact.
4. Add cohort-aware freshness/packing, one-time metrics and state persistence.
5. Resolve config and perform a capped SWE launch sanity check before the two full runs; no separate calibration or pilot training campaign.

Targeted additions to existing algorithm/advantage/config tests are justified for: p=0 equivalence; all-fail retry/give-up; fresh all-success filter; mixed-history acceptance; historical counts after expiration; anchored sum and scale; zero-negative case; independent visit IDs; incomplete-round exclusion; replayed-round deduplication. Test pure transformations and state transitions, not a heavily mocked end-to-end dispatcher. Use a capped real integration check for scheduling, staleness and restart behavior.

Acceptance: no duplicate episode accounting; no expired payload reaches training; deterministic continuation decisions from persisted RNG; no task/visit mixing; valid cohorts have zero-sum sample advantages; active sampling still replenishes batches; p=0 matches static GRPO credit and filtering on valid binary cohorts. For multi-turn SWE, preserve every turn’s behavior logprobs and router replay data and reset the sandbox on every retry.
