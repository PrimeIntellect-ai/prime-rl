# Staleness and pacing

How the async RL pipeline bounds off-policy staleness, when dispatch starts and
stops, and why shutdown is race-free. Two counters govern everything:

- `N` — the batch the orchestrator is collecting (`progress.step`, 1-indexed).
- `v` — the trainer's policy version (0-indexed). `v` advances when the trainer
  *publishes* a version; inference applies it immediately and pauses generation
  during the update, so published and applied versions are indistinguishable to
  every consumer.

A rollout is stamped with the version it was generated from (`v_gen`). If it is
consumed in batch `N`, it trains `lag = (N-1) - v_gen` versions behind — batch
`N` produces `v{N}` from `v{N-1}`, so `lag = 0` is fully on-policy.

## The three rules

Each rule bounds `N - v` (or a rollout's `lag`) at one lifecycle point. They are
independent: each one prevents a failure mode the others cannot.

| Rule | Bound | Enforced at | Failure mode it prevents |
|---|---|---|---|
| **START** | dispatch pauses while `(N-1) - v > TARGET_LAG` | dispatch gate | generating data that is born stale |
| **ADVANCE** | batch `N` ships only once `v ≥ N-1-TARGET_LAG` | ship hold | the batch counter outrunning the trainer |
| **DIE** | rollouts that straddle more than `max_off_policy_steps` weight updates in flight are cancelled | weight-update hook | long-running stragglers aging without bound |

With `TARGET_LAG = 1`, START and ADVANCE together keep consumption lag at ≤ 2
in steady state (≤ 3 with batch-boundary spill); DIE only fires on rollouts
whose own duration spans many trainer steps (agentic long tails).

## The two worlds

Which rule binds depends on the ratio of trainer step time `T` to per-batch
generation time `G`. The rules never change — only which one saturates:

**World 1 — generation-bound (`T < G`, the trainer waits).** `v` catches up
after every ship, so `(N-1) - v ≤ 1` always: START and ADVANCE never bind and
DIE stays quiet. The pipeline is paced by generation; the trainer idles between
batches. All staleness rules are inert insurance.

**World 2 — trainer-bound (`T > G`, inference races).** Batches fill faster
than versions arrive. ADVANCE binds every step (the ship hold paces `N` to
`v`); START closes dispatch between version publishes (inference alternates
generate/idle, idle fraction `1 - G/T`). Without ADVANCE this world is where
the orchestrator finishes all its batches from buffered rollouts, exits, and
strands the trainer (see below).

The crossover is smooth: the same predicates are evaluated everywhere, and no
configuration switch distinguishes the worlds.

## Shutdown correctness

For in-memory weight transports (NCCL, NIXL) the trainer *blocks* inside each
broadcast until the orchestrator's weight watcher completes the matching apply.
Two guarantees make teardown race-free:

1. **ADVANCE is the liveness guarantee.** The orchestrator cannot exit before
   shipping its final batch `M`, which requires `v ≥ M-2` — and `v{M-2}` is
   exactly the last version whose broadcast needs a live watcher (the trainer
   skips the final `TARGET_LAG + 1` in-memory broadcasts because their receiver
   is torn down; this is also why the hold never waits for an unpublishable
   version).
2. **The watcher drains before dying.** `WeightWatcher.stop()` waits for an
   in-flight apply to complete before cancelling, so a shutdown that races the
   last handshake finishes it instead of stranding the trainer. The
   orchestrator's global teardown budget bounds this wait.

No other component participates in shutdown ordering: the trainer only ever
waits on the watcher, and the watcher outlives every wait.

## Who starts and stops dispatch

Nobody signals — dispatch is *pulled*. The dispatcher re-checks the gate before
each scheduling decision, so stop/resume is a stateless predicate over
`(N, v)`, re-evaluated on its own loop. The single push-style signal in the
system is the version-advance event that wakes a held ship. Throughput is
identical to an unpaced orchestrator in both worlds: in World 1 no rule ever
binds, and in World 2 the trainer is the bottleneck regardless — the hold only
changes *where* a finished batch waits (in the orchestrator instead of the
trainer's queue), never when the trainer gets to consume it.
