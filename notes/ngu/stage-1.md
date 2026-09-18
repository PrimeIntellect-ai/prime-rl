# Stage 1: NGU implementation

Binary NGU is implemented in `orchestrator/ngu.py`, the NGU algorithm, TrainSource and TrainSink. See [configuration and semantics](../../docs/algorithms.md#never-give-up-ngu) and [cluster verification](smoke.md).

## Ownership

The dispatcher still handles ordinary K-sized physical groups. Every retry receives a new group ID and current dispatch step, and starts fresh episodes of the same task. Scheduling does not run inside the environment or trainer.

`TrainSource` owns one `NGUController` per NGU source. A logical visit spans physical rounds and owns its task, unique visit ID, historical attempt/success counts and retained episodes. Independent visits to the same task have separate histories. Queued retries run before fresh source selection; source ratios therefore describe fresh selections rather than compute shares.

`TrainSink` commits completed rounds to the controller. An all-zero round either queues a retry with probability p or ends the visit. A round with a positive reward returns its retained history for scoring. Errors, missing trainable payloads and cancellations terminate the visit without converting those attempts into zeros.

`NGUAlgorithm` computes the historical centered binary baseline and anchors positive advantages. The sink uses the existing routing, trace conversion and trainer transport. No changes to the trainer loss, model numerical dtypes, or dispatcher are required.

## Correctness and limits

- Live-policy, single-agent episodes with one trainable trace and rewards exactly 0 or 1. Binary rewards are checked at runtime. No length penalty.
- Historical counts survive payload expiry. Retention uses inclusive age `(training_step - 1) - policy.start`, bounded by both NGU history age and the global off-policy limit.
- Retained positives get `1-S/C`; negatives are rescaled to center the retained cohort. The sink checks freshness again before shipping and recomputes advantages after removing stale payloads. A cohort without both reward classes is discarded.
- Explicit `preserve_groups=true` ships whole cohorts, allowing the last cohort to exceed the trace/token batch target. Both experiment arms enable it; the default for other runs remains false.
- `max_history_tokens` bounds retained graph tokens across unsuccessful visits per source. Oldest payloads are evicted while counts remain. This is not a global memory limit: finalized cohorts and pending metric windows are outside the budget. Dispatcher concurrency bounds physical work; there is no fixed retry-count cap.
- A configurable no-output guard counts unsuccessful retry rounds. The default remains ten batch equivalents; both SWE arms use 100.
- Generation episodes are logged once. Later shipment annotations record visit ID, round, historical counts and advantage without duplicating episode records.

## Checkpoint and resume

The checkpoint manager snapshots source and sink state on the orchestrator event loop. It saves the continuation RNG, source sampler, committed visit histories and queued training cohorts. Wire episodes serialize to data dictionaries with float rounding disabled, retaining behavior logprobs, advantages and training arrays.

On resume, unfinished physical rounds restart with new group IDs. Only completed rounds contributed historical counts, so partial results are not counted twice. Queued cohorts retain their identities and are rechecked for freshness. This restores committed NGU state; it does not reproduce in-flight generations or exact asynchronous ordering.

## Verification

Pure tests cover anchoring, expiry, visit isolation, give-up, incomplete rounds, eviction, configuration validation, RNG restoration and lossless wire checkpoint round trips. The small real reverse-text cluster run exercises retries, expanded cohorts, batch overshoot, training, checkpoint saving and resume. The six-node SWE experiment remains a separate deployment.
