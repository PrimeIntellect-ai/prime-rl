# RL Pipeline

This page is the design reference for the orchestrator's RL pipeline: its goals, its components and their interfaces, the invariants it targets, and where today's code falls short of them. The trainer, inference servers, and env servers appear only as endpoints: what they receive and what they send.

If you change the pipeline, check the change against [Invariants](#invariants), [Known Gaps](#known-gaps), and the [Checklist for Changes](#checklist-for-changes).

## Table of Contents

- [Overview](#overview)
- [Terms](#terms)
- [Goals](#goals)
- [Invariants](#invariants)
- [Steps, Versions, and Staleness](#steps-versions-and-staleness)
- [Weight Updates](#weight-updates)
- [Throttles](#throttles)
  - [Engine Load](#engine-load)
  - [Two Checks for Run-Ahead](#two-checks-for-run-ahead)
  - [Why Run-Ahead and Staleness Are Separate](#why-run-ahead-and-staleness-are-separate)
  - [Why Emitting Never Blocks](#why-emitting-never-blocks)
- [Life of an Episode](#life-of-an-episode)
- [Known Gaps](#known-gaps)
- [Checklist for Changes](#checklist-for-changes)

## Overview

The orchestrator is one process running three concurrent loops on one asyncio event loop: the dispatcher, the main loop, and the weight watcher. The trainer, inference servers, and env servers are separate processes.

```
                tasks
                  |
                  v
      +-------------------+   results   +-------------------+   batch N   +-----------+
      |    dispatcher     |------------>|     main loop     |------------>|  trainer  |
      |  starts episodes  |             |  groups, scores,  |             +-----------+
      +-------------------+             |  ships batches    |                   |
          |          ^                  +-------------------+                   |
          |          |                            ^                             |
 episodes |          | fence, cancel,             | wakes                       | weights
          |          | reopens gate               | ship hold                   | v{N}
          v          |                            |                             |
   +-------------+   |       +-----------------------------------+              |
   | env servers |   +-------|              watcher              |<-------------+
   |      +      |           |   loads new weights, advances     |
   |  inference  |<----------|   the policy version              |
   +-------------+   loads   +-----------------------------------+
                     v{N}
```

| Component | Receives | Emits | Decides |
|---|---|---|---|
| Train source | Requests for a task | Tasks | Which env and task come next |
| Dispatcher | Tasks, throttle signals | Episode requests; one result per attempt | When an episode may start |
| Env servers + inference | Episode requests | Finished episodes | Nothing the pipeline controls |
| Main loop | Results | Batches to the trainer | When a ready batch may ship |
| Train sink | Results, from the main loop | Ready batches | Which groups and traces train |
| Batch packer | A ready batch | Micro batches per trainer rank | How samples are split across ranks |
| Trainer | Batches | New weights | Nothing the pipeline controls |
| Weight watcher | New weights | Weights to inference; version signal | When to swap weights |
| Concurrency controller | Inference load metrics | The in-flight cap | How many episodes may run at once |

## Terms

- **Episode** (also called a rollout): one run of a task through an env server.
- **Trace**: one trainable token sequence inside an episode. An episode can contain several. `batch_size` counts traces.
- **Group**: the `group_size` episodes of one task, scored relative to each other.
- **Attempt**: one started episode, whether it finishes, fails, or is cancelled.
- **Result**: what the dispatcher reports for an attempt: a finished episode, a failure, or one cancellation covering everything a dropped group still owes. Results reach the main loop through a queue.
- **Curriculum**: an env's task sampler, which also decides whether a finished group trains. See [Algorithms](algorithms.md#curricula).
- **Step and policy version**: steps count batches from 1; policy versions count from 0, with `v0` the base model. Batch $N$ trains on `v{N-1}` and produces `v{N}`.
- **Dispatch version**: the policy version inference served when a group's first episode started. Every episode in the group carries it.
- **Staleness**: for a rollout trained in batch $N$ with dispatch version `v{k}`, the number $(N-1) - k$.
- **Run-ahead**: how many versions the policy a batch trains on may be ahead of the policy inference serves when that batch ships.

## Goals

The pipeline optimizes for three things:

- Keep inference saturated with useful work.
- Keep staleness bounded, and as low as throughput allows.
- Never hang silently, and never train on data past the staleness bound.

It accepts two trade-offs to get there.

**Run-ahead of exactly one step.** Run-ahead is fixed at 1 (the `TARGET_LAG` constant). A run-ahead of zero is synchronous RL: inference idles during every trainer step. Above one mostly adds staleness. In the vast majority of RL workloads inference is the bottleneck: the trainer finishes its step first, so a larger lead is never reached. When inference is faster, it races to the larger lead and waits there, so the data is staler with no throughput gain. A larger lead helps only to absorb variation in step times.

**Bounded staleness.** No rollout trains more than `max_off_policy_steps` versions behind the policy it trains; staler rollouts are dropped. A higher value drops fewer long rollouts (more throughput) but trains on staler tokens (less stable). A lower value is more stable but drops more rollouts and biases training toward rollouts that finish quickly.

## Invariants

Each invariant is a guarantee the pipeline targets, followed by the component that enforces it. Where today's code falls short, the shortfall is listed under [Known Gaps](#known-gaps).

1. **Every opened train group reaches exactly one terminal state**: its traces are queued for training, or it is dropped for having no trainable traces, rejected by the curriculum, voided as stale, or abandoned. Enforced by the dispatcher, which reports every attempt exactly once, and the train sink, which counts each group to completion.
2. **No trained token was generated by a policy more than `max_off_policy_steps` versions behind the policy it trains.** Staleness is measured from the dispatch version, which can only overstate it. Enforced by the train sink, before every batch cut.
3. **Batch $N$ ships only once inference serves at least `v{N-2}`**, that is, `v{N-1}` minus the run-ahead. Enforced by the main loop's ship hold.
4. **A new episode starts only while in-flight episodes are below the current cap, and the cap stays within its configured bounds.** Lowering the cap cancels nothing by itself, so in-flight episodes can briefly exceed it. Enforced by the dispatcher and the concurrency controller.
5. **No new episode starts while a weight update is in progress.** Episodes already running continue across it, so a multi-turn episode can span versions; staleness accounts for that. Enforced by the dispatcher.
6. **Every trainer data-parallel rank receives the same number of micro batches per step, with the same modality at each index**, so the ranks' collectives stay in lockstep. Enforced by the batch packer.
7. **The orchestrator stays alive through the trainer's final weight broadcast**, so the trainer is not stranded inside the broadcast handshake. Enforced by the orchestrator at shutdown.
8. **Liveness: every wait on another component is bounded or fails when that component dies, and nothing on the weight-update path waits on the main loop.** This backs the "never hang silently" goal. Enforced by the main loop, which fails when the dispatcher or weight watcher loop dies.
9. **Emitting a result never blocks; back-pressure acts only on starting new work.** This keeps weight updates from waiting on the main loop, as liveness requires. Enforced by the dispatcher.
10. **Each eval result measures one policy version.** Not enforced today.

## Steps, Versions, and Staleness

The step being collected advances right after a batch ships.

```
batch (step)                      1      2      3      4      5
trains on                         v0     v1     v2     v3     v4
ships once inference has          any    v0     v1     v2     v3     (run-ahead 1)
```

Staleness counts both generation time (a long rollout can span several weight updates) and queue time:

```
policy served by inference      v3         v4         v5         v6
                            ----+----------+----------+----------+------->
group dispatched at v3          [====== generating ======]
                                                         [== queued ==]
                                                                       ships in batch 7, trains on v6

staleness = 6 - 3 = 3     in flight: v5 - v3 = 2     in queue: 1
```

The in-queue share includes the run-ahead itself: a rollout shipped the moment it finishes can still be one version stale. Three consequences:

- A group's episodes share one dispatch version, so they age out together. A member started after a weight update still carries the group's older version, so measured staleness is an upper bound on how stale its tokens really are.
- Episodes from frozen-model sources (`algo.sampling.source` other than `"policy"`) never go stale: their generator does not change with policy updates.
- A rollout is past the bound when its staleness exceeds `max_off_policy_steps`.

## Weight Updates

A weight update advances the policy version. Several throttles act at fixed points in it:

```
trainer           publishes v{N}
weight watcher    notices v{N}, notifies the dispatcher
dispatcher        [fence]  stops new starts; a start already in progress finishes
dispatcher        [cancel] drops train groups that can no longer train within the bound
inference         loads v{N}
weight watcher    advances the policy version to N
dispatcher        lifts the [fence]
orchestrator      may start an eval epoch, re-checks the [gate], wakes the [hold]
```

The cancel runs before inference pauses for the swap, so the resulting request aborts settle while the engines are still stepping. Episodes already in flight keep running across the swap.

## Throttles

Every throttle protects one thing. The tags match the [Life of an Episode](#life-of-an-episode) diagram.

| Throttle | Protects | Acts on | Enforced by |
|---|---|---|---|
| `[cap]` adaptive in-flight cap | Inference KV cache | New starts | Dispatcher, concurrency controller |
| `[burst]` admission burst cap | Inference prefill bursts | New starts | Dispatcher |
| `[rate]` `tasks_per_minute` | External services | New starts | Dispatcher |
| `[gate]` dispatch gate | Run-ahead (early check) | New train starts | Orchestrator |
| `[hold]` ship hold | Run-ahead (guarantee) | Batch shipping | Main loop |
| `[cancel]` early stale cancel | Inference compute | In-flight groups | Dispatcher, at weight updates |
| `[sweep]` staleness sweep | Staleness (guarantee) | Queued traces | Train sink |
| `[fence]` weight-update fence | Weight updates | New starts | Dispatcher, at weight updates |
| `[backlog]` output back-pressure | Orchestrator memory | New starts | Dispatcher |

### Engine Load

The concurrency controller adapts the cap to inference load. It grows the cap slowly while the engines have headroom, trims it under KV-cache pressure, and on overload (preemptions or a persistent queue for KV capacity) cuts it and cancels the youngest groups first, since they have the least inference spent. Setting the minimum and maximum equal gives fixed concurrency. Train and eval share the cap. The burst cap limits net growth per time window, so a raised cap does not land a wall of prefills at once.

### Two Checks for Run-Ahead

Both checks test the same condition: batch $N$ may proceed once inference serves at least `v{N-2}`.

- **Dispatch gate** (early check). While the condition fails, the dispatcher starts no new train episodes. Eval ignores the gate: eval results never train, so run-ahead does not apply to them.
- **Ship hold** (guarantee). The main loop holds a ready batch until the condition holds. The gate alone is not enough, because rollouts that already finished can fill a batch while the gate is closed.

### Why Run-Ahead and Staleness Are Separate

The run-ahead checks bound the staleness every rollout picks up from the pipeline itself. The staleness bound catches long rollouts that span many weight updates. Batches form greedily from whatever finishes first, regardless of submission order, so the two concerns are independent and need separate knobs.

The in-flight cap is not a run-ahead bound. It does cap how much work can finish behind a closed gate, and that backlog adds queue staleness.

### Why Emitting Never Blocks

The dispatcher never waits when it reports a result: the result queue is unbounded. Instead, it starts no new episodes, train or eval, while the queue holds at least the configured maximum cap (`concurrency.max_inflight`) of results. Starts resume once the main loop drains the queue below that threshold.

Blocking on emit would deadlock. During a weight update, the early cancel reports cancellations to the result queue, while the main loop, the queue's only reader, may itself be in the ship hold waiting for that update to finish.

The threshold is the configured maximum, not the current cap. A low current cap would shrink the buffer and idle inference during routine short main-loop stalls. During a stall, memory is bounded by the threshold's worth of queued results plus the episodes still in flight when starts pause. When `concurrency.max_inflight` is unset, nothing bounds the backlog.

## Life of an Episode

From task to trainer, with each throttle tagged where it acts:

```
train source      picks an env by its configured ratio, draws a task from its curriculum
     |
     v
dispatcher        [cap] [burst] [rate] [gate] [fence] [backlog]
     |            starts the group's episodes one at a time
     v
env server        runs the episode, calling inference for each model turn
     |
     v
result queue      finished episode | failure | group cancellation
     |
     v
main loop         hands each result to the train sink
     |
     v
train sink        waits for every attempt in the group, scores the group,
     |            lets the curriculum admit or reject it, converts traces to samples
     |            [sweep] voids queued traces past the staleness bound
     v  (batch_size traces queued)
main loop         [hold] waits until inference serves v{N-2}
     |            batch packer splits the batch per trainer rank, then it is sent
     v
trainer           trains batch N on v{N-1}, publishes v{N}
     |
     v
weight watcher    [fence] [cancel], then loads v{N} (see Weight Updates)
```

Each group ends in one terminal state: its traces are queued for training, or it has no trainable traces, the curriculum rejects it, the staleness bound voids it (in flight or queued), or the dispatcher abandons it after an overload cut.

The main loop is the only reader of the result queue. It routes each result to the train or eval sink, ships a batch when one is ready, advances the step, and saves checkpoints. It does slow work inline (scoring, packing, sending, the ship hold), so while it is busy, results pile up until `[backlog]` pauses new starts. Checkpoint saves run synchronously and briefly block the whole event loop.

## Known Gaps

Places where today's code does not fully meet an invariant above, or a guarantee a reader would reasonably expect. Unlike the rest of the page, entries name the code at fault, so they can be acted on. Fixing a gap means deleting its entry in the same change.

- **The ship hold can wait forever** (liveness). `Orchestrator.finalize_train_batch` waits on `version_advanced` with no timeout and without calling `_raise_if_component_stopped`. If the watcher task dies during a hold, `main_loop` hangs silently.
- **A dead trainer goes unnoticed** (liveness). Within the orchestrator, nothing detects that the trainer has exited: the watcher keeps polling for the next version, the dispatch gate stays closed, and `main_loop` idles without error.
- **Some failures are only logged or lost** (liveness). `WeightWatcher.apply_policy_update` logs exceptions from `on_version_pending` and `on_new_version` as warnings and continues. `Dispatcher.cancel_inflight` starts its cancellation with `asyncio.create_task` and keeps no reference, so an exception there is never seen.
- **The rate limiter can stall a weight update** (liveness). `Dispatcher.acquire` waits on the `tasks_per_minute` limiter while `fill_inflight` holds `scheduling_lock`, and `on_version_pending` waits for that lock, so an update can wait up to one rate-limit interval.
- **The final-broadcast wait gives up** (stays alive through the final broadcast). `Orchestrator.wait_for_version` stops waiting after `weight_broadcast.timeout` seconds and proceeds to teardown, which can leave the trainer inside the handshake.
- **Wind-down cancellations skip accounting** (every group reaches one terminal state). `Dispatcher.cancel_inflight_train_episodes` (called by `Orchestrator.start_draining` once the step budget is reached) and `Dispatcher.cancel_inflight_episodes` (shutdown) cancel episodes without putting anything on `out_q`. This is harmless today because no train batch ships while draining.
- **A self-cancelled episode is never accounted for** (every group reaches one terminal state). If an episode task ends in a `CancelledError` the dispatcher did not start, `Dispatcher.handle_completed_request` returns without emitting to `out_q`. The group never completes in `TrainSink`, its arrived episodes stay buffered, and its `GroupState` stays in `Dispatcher.groups`.
- **Eval results can mix policy versions** (each eval result measures one version). Eval episodes ignore the stale cancel at weight updates, so an eval epoch that outlasts a trainer step spans several versions: members started after an update, and later turns of multi-turn episodes, run on newer weights. `EvalSink` does not read the episodes' policy versions, so the mix goes unflagged.
- **A crashed run still saves a checkpoint.** The `finally` block in `Orchestrator.start` saves an orchestrator checkpoint at the last shipped step even when `main_loop` raised. That step need not have a matching trainer checkpoint.

## Checklist for Changes

Before merging a change to the dispatcher, the main loop or batch shipping, the train sink, concurrency control, the weight watcher, batch transports, or the trainer's data intake:

- Walk the [invariants](#invariants) and name the ones the change could affect. If it changes one on purpose, update this page in the same change.
- If the change fixes a [known gap](#known-gaps), delete the entry; if it finds a new one, add it.
- Put new "start new work?" throttles in the dispatcher's start check, next to the existing ones, not on the result path. Reporting results must never block: weight updates report results while the main loop may be waiting on that update.
- Slow work added to the main loop delays all result processing, including group scoring and batch cuts. Run CPU-heavy work off the event loop.
- Anything that waits on another component should fail loudly if that component dies, rather than wait forever.
