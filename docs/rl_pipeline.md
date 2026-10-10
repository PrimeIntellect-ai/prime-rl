# RL Pipeline

This page describes how the orchestrator moves work from tasks to the trainer, what each throttle protects, and which invariants the system guarantees, each tied to the code that enforces it. The trainer, inference servers, env servers, and transports appear only as endpoints: what they receive and what they send.

If you change the orchestrator, check the change against [Invariants](#invariants), [Known Gaps](#known-gaps), and the [Checklist for Changes](#checklist-for-changes).

## Table of Contents

- [Goals](#goals)
- [Components and Data Path](#components-and-data-path)
- [Steps, Versions, and Staleness](#steps-versions-and-staleness)
- [Throttles](#throttles)
  - [Engine Load](#engine-load)
  - [External Limits](#external-limits)
  - [Run-Ahead: the Pause Rule](#run-ahead-the-pause-rule)
  - [Staleness: the Drop Rule](#staleness-the-drop-rule)
  - [Why Run-Ahead and Staleness Are Separate](#why-run-ahead-and-staleness-are-separate)
  - [Output Back-Pressure](#output-back-pressure)
  - [Weight-Swap Sequence](#weight-swap-sequence)
- [Invariants](#invariants)
- [Known Gaps](#known-gaps)
- [Checklist for Changes](#checklist-for-changes)

## Goals

The pipeline optimizes for three things:

- Keep inference saturated with useful work.
- Keep off-policy staleness bounded, and as low as throughput allows. Staleness is how many policy versions separate the policy a rollout was generated with from the policy it trains (defined precisely in [Steps, Versions, and Staleness](#steps-versions-and-staleness)).
- Never hang silently, and never train on data past the staleness bound.

It accepts two trade-offs to get there.

**Run-ahead of exactly one step.** Run-ahead is how many versions the policy a batch trains on may be ahead of the policy inference serves when that batch ships. It is the constant `TARGET_LAG = 1` in `orchestrator/orchestrator.py`. A run-ahead of zero is synchronous RL: inference idles during every trainer step. Above one mostly adds staleness. If the trainer is faster than inference, the extra lead is never reached. If inference is faster, it races to the larger lead and waits there, so the data is staler with no throughput gain. A larger lead helps only to absorb variation in step times.

**`max_off_policy_steps` (default 8).** Rollouts staler than this bound are dropped. A higher value drops fewer long rollouts (more throughput) but trains on staler tokens (less stable). A lower value is more stable but drops more rollouts and biases training toward rollouts that finish quickly.

## Components and Data Path

Four kinds of process take part:

- **Orchestrator**: one asyncio event loop. Everything on this page runs here.
- **Trainer**: a synchronous torch loop.
- **Inference servers**: vLLM, serving the current policy.
- **Env servers**: run episodes, calling inference for each model turn.

Terms used throughout:

- **Episode** (also called a rollout): one run of a task through an env server.
- **Trace**: one trainable token sequence inside an episode. An episode can contain several. `batch_size` counts traces.
- **Group**: the `group_size` episodes of one task, scored relative to each other.
- **Attempt**: one scheduled episode, whether it finishes, fails, or is cancelled.
- **Curriculum**: an env's task sampler, which also decides whether a finished group trains. See [Algorithms](algorithms.md#curricula).

One training step flows through the orchestrator like this. Bracketed tags mark where each [throttle](#throttles) acts.

```
TrainSource.next_task            pick a task through the env's curriculum
      |
      v
Dispatcher.fill_inflight         [cap] [burst] [rate] [gate] [fence] [backlog]
      |                          start one asyncio task per episode
      v
episode tasks                    env server <-> inference servers
      |
      v
Dispatcher.out_q (unbounded)     episode | DispatchFailure | GroupCancellation
      |
      v
Orchestrator.main_loop
      |
      v
TrainSink.add / fail / cancel    finalize group, score it, curriculum admit,
      |                          convert traces to TrainingSample
      |                          [sweep] void queued traces past the bound
      v  (batch_size traces queued)
finalize_train_batch             [hold] wait for policy v{N-1-TARGET_LAG}
      |                          BatchPacker.pack, BatchSender.send
      v
trainer                          DataLoader.get_batch, optimizer step,
      |                          broadcast weights v{N}
      v
WeightWatcher                    [fence] [cancel] then load v{N} into inference,
      |                          advance policy.version, re-check [gate] and [hold]
      +--> inference serves v{N}
```

$N$ is the batch being shipped, and `v{N}` is the policy the trainer produces from it; [Steps, Versions, and Staleness](#steps-versions-and-staleness) defines both.

Details at each stage:

- `TrainSource.next_task` picks an env by its configured ratio and draws a task from that env's curriculum.
- The `Dispatcher` starts a group's episodes one at a time, each as its own asyncio task, and every finished attempt goes onto `Dispatcher.out_q`: a native episode, a `DispatchFailure` when no episode was produced, or one `GroupCancellation` covering everything a dropped group still owes.
- `TrainSink` collects a group until every member is accounted for, scores it with the env's algorithm, asks the curriculum whether to admit it (`TrainSource.on_result`), and converts admitted traces to `TrainingSample`s (`transports/batch/types.py`). Once `batch_size` traces are queued, it cuts a `TrainBatch`.
- `BatchPacker.pack` (`orchestrator/batch.py`) packs the samples into one list of micro batches per trainer data-parallel (DP) rank, and a `BatchSender` (filesystem or ZMQ, `transports/batch/`) sends them. The trainer reads one step's micro batches with `DataLoader.wait_for_batch` and `DataLoader.get_batch` (`trainer/rl/data.py`), takes the optimizer step, and broadcasts the new weights.
- `WeightWatcher` (`orchestrator/watcher.py`) polls for a newly published version and applies it, as shown in [the weight-swap sequence](#weight-swap-sequence).

The orchestrator's asyncio tasks:

- `Orchestrator.main_loop` runs inline in the start task. It awaits slow work directly (group scoring, packing, sending, the ship hold), so it can stop consuming `out_q` for a while; results then accumulate in `out_q` until `[backlog]` pauses new starts. Checkpoint saves run synchronously and briefly block the whole event loop. CPU-heavy steps (packing, trace conversion) run in threads via `asyncio.to_thread`.
- `Dispatcher.start` and `WeightWatcher.start` run as background tasks, alongside the inference-metrics poller that feeds `ConcurrencyController`, the event-loop lag monitor, and the periodic logger.

## Steps, Versions, and Staleness

Steps are 1-indexed; policy versions are 0-indexed, with `v0` the base model. Batch $N$ trains on `v{N-1}` and produces `v{N}`. `progress.step` is the batch currently being collected; it advances right after a batch ships.

```
batch (step)                      1      2      3      4      5
trains on                         v0     v1     v2     v3     v4
ships once inference has          any    v0     v1     v2     v3     (TARGET_LAG = 1)
```

The staleness of a rollout trained in batch $N$ is $(N-1) - k$, where `v{k}` is its group's dispatch version: the policy version when the group's first episode was scheduled (`GroupState.policy_version_at_start`). It counts both generation time (a long rollout can span several weight updates) and queue time:

```
policy served by inference      v3         v4         v5         v6
                            ----+----------+----------+----------+------->
group dispatched at v3          [====== generating ======]
                                                         [== queued ==]
                                                                       ships in batch 7, trains on v6

staleness = 6 - 3 = 3     in flight: v5 - v3 = 2     in queue: 1
```

`episode_staleness` and `min_fresh_version` (`orchestrator/utils.py`) compute these quantities. The in-queue share includes the run-ahead itself: a rollout shipped the moment it finishes can still be up to `TARGET_LAG` versions stale. Three consequences:

- A group's episodes share one dispatch version, so they age out together. A member scheduled after a weight swap still carries the group's older version, so measured staleness is an upper bound on how stale its tokens really are.
- Episodes from frozen-model sources (`algo.sampling.source` other than `"policy"`) never go stale: their generator does not change with policy updates.
- A rollout is past the bound when its staleness exceeds `max_off_policy_steps`, that is, when its dispatch version is older than `min_fresh_version(progress.step, max_off_policy_steps)`.

## Throttles

Every throttle protects one thing. Two of them use `TARGET_LAG`.

| Throttle | Protects | Acts on | Enforced in |
|---|---|---|---|
| `[cap]` adaptive in-flight cap | Inference KV cache | New starts | `fill_inflight`, `ConcurrencyController` |
| `[burst]` admission burst cap | Inference prefill bursts | New starts | `Dispatcher.admission_budget` |
| `[rate]` `tasks_per_minute` | External services | New starts | `Dispatcher.acquire` |
| `[gate]` dispatch gate | Run-ahead (early check) | New train starts | `Orchestrator.update_dispatch_gate` |
| `[hold]` ship hold | Run-ahead (guarantee) | Batch shipping | `Orchestrator.finalize_train_batch` |
| `[cancel]` early stale cancel | Inference compute | In-flight groups | `Dispatcher.on_version_pending` |
| `[sweep]` staleness sweep | Staleness (guarantee) | Queued traces | `TrainSink._drop_stale` |
| `[fence]` weight-swap fence | Weight swaps | New starts | `Dispatcher.on_version_pending` |
| `[backlog]` output back-pressure | Orchestrator memory | New starts | `Dispatcher.fill_inflight` |

### Engine Load

`ConcurrencyController` (`orchestrator/concurrency.py`) moves the dispatcher's in-flight cap based on vLLM KV-cache usage, preemptions, and requests queued for KV capacity. It grows the cap slowly and multiplicatively while the engines are clear and in-flight episodes are near the cap. Above a soft KV-usage threshold it trims the cap and cancels nothing; above a hard threshold it also cancels the excess, youngest groups first. On preemptions or a persistent capacity queue it cuts the cap and cancels the excess.

Mind the naming. The config field `concurrency.max_inflight` is the ceiling; the dispatcher attribute `Dispatcher.max_inflight` is the current cap, which stays within `[concurrency.min_inflight, concurrency.max_inflight]`. Setting `min_inflight = max_inflight` gives fixed concurrency. The ceiling also sets the `[backlog]` threshold. One permit is one episode, and train and eval share the permits.

`Dispatcher.admission_budget` limits how fast the pool grows: per 5-second window, net new admissions are capped at the larger of the biggest train `group_size` and a tenth of the current cap. Replacing an episode that finished naturally is free.

### External Limits

`tasks_per_minute` rate-limits episode starts with an `AsyncLimiter`. Use it for sandbox-backed envs whose provider limits request rate.

### Run-Ahead: the Pause Rule

The orchestrator may work on batch $N$ only once inference has `v{N-1-TARGET_LAG}`. The same inequality, $(N-1) - \text{policy.version} \le$ `TARGET_LAG`, is checked at two points:

- **Dispatch gate** (early check). `Orchestrator.update_dispatch_gate` closes the gate (clears the `Dispatcher.dispatch_allowed` event) or opens it (sets the event) after each ship and each policy update. While the gate is closed, `fill_inflight` starts no new train episodes. Eval ignores the gate: eval results never train, so run-ahead does not apply to them.
- **Ship hold** (guarantee). `Orchestrator.finalize_train_batch` waits before sending batch $N$ until `policy.version` is at least $N - 1 -$ `TARGET_LAG`. The gate alone is not enough, because rollouts that already finished can fill a batch while the gate is closed.

### Staleness: the Drop Rule

Rollouts past `max_off_policy_steps` are dropped at two points:

- **Early cancel** (saves compute). `Dispatcher.on_version_pending` drops in-flight train groups that can no longer train within the bound.
- **Staleness sweep** (guarantee). `TrainSink._drop_stale` voids queued traces past the bound before every batch cut, and checks each newly inserted group on arrival.

### Why Run-Ahead and Staleness Are Separate

The pause rule bounds the staleness every rollout picks up from the pipeline itself. The drop rule bounds long rollouts that span many weight updates. Batches form greedily from whatever finishes first, regardless of submission order, so the two concerns are independent and need separate knobs.

The in-flight cap is not a run-ahead bound. It does cap how much work can finish behind a closed gate, and that backlog adds queue staleness.

### Output Back-Pressure

Emitting a result never blocks: every emit to `Dispatcher.out_q` uses `put_nowait`, and the queue is unbounded. Instead, `fill_inflight` starts no new episodes, train or eval, while `Dispatcher.is_out_q_backlogged`: `out_q` holds at least `concurrency.max_inflight` results. Starts resume once `main_loop` drains the queue below that threshold.

Blocking on emit would deadlock. During a [weight swap](#weight-swap-sequence), the early cancel emits `GroupCancellation`s to `out_q`, while `main_loop`, the only consumer, may itself be in the ship hold waiting for that swap to finish.

The threshold is the configured ceiling, not the current cap. A low current cap would shrink the buffer and idle inference during routine short `main_loop` stalls (packing, sending, checkpoint saves). During a stall, memory is bounded by the ceiling's worth of queued results plus the episodes still in flight when starts pause. When `concurrency.max_inflight` is unset, nothing bounds the backlog.

### Weight-Swap Sequence

`WeightWatcher.apply_policy_update` runs this sequence for each new version:

```
receiver.wait_published(N)            trainer has published v{N}
Dispatcher.on_version_pending(N)
    policy_update_pending = True      [fence] fill_inflight starts nothing new
    acquire + release scheduling_lock wait out a scheduling call already running
    drop_group(..., reason="stale")   [cancel] live train groups past the bound
receiver.receive(N)                   inference loads v{N}
policy.version = N
Dispatcher.on_new_version(N)          policy_update_pending = False
update hooks
    trigger_eval                      maybe start an eval epoch
    on_policy_update                  update_dispatch_gate, wake the ship hold
```

The early cancel runs before inference pauses for the swap, so the resulting request aborts settle while the engines are still stepping.

## Invariants

Each item states the guarantee, then the code that enforces it. Where today's code falls short of a guarantee, the shortfall is listed under [Known Gaps](#known-gaps).

1. **Every opened train group reaches exactly one terminal state in `TrainSink`**: its traces are queued for training, or it is rejected by the curriculum, voided as stale, or abandoned. The mechanism: while the run is active, each attempt reaches `out_q` exactly once (as an episode, a `DispatchFailure`, or under its group's `GroupCancellation`), so `TrainSink` can count every group to `group_size`. Enforced by `Dispatcher.handle_completed_request`, `emit_episode`, and `drop_group`.
2. **No trained token was generated by a policy more than `max_off_policy_steps` versions behind the policy it trains.** The check uses the group's dispatch version, a conservative stand-in that can only overstate a token's staleness. `TrainSink._drop_stale` sweeps queued traces before each batch cut and checks each inserted group; a stale `GroupCancellation` also voids its group's already-arrived episodes (`TrainSink.process_group`).
3. **Batch $N$ ships only once `policy.version` $\ge N - 1 -$ `TARGET_LAG`.** `Orchestrator.finalize_train_batch`.
4. **A new episode starts only while in-flight episodes are below the current cap, and the cap stays within `[concurrency.min_inflight, concurrency.max_inflight]`.** `Dispatcher.fill_inflight` checks `available_permits`; `ConcurrencyController.clamp` bounds the cap, with no ceiling when `concurrency.max_inflight` is unset. Lowering the cap cancels nothing on its own (`Dispatcher.set_limit`), so in-flight episodes can exceed a freshly lowered cap until they finish; only hard trims and cuts cancel the excess (`ConcurrencyController.resize_down`).
5. **No new episode starts while a weight swap is in progress.** Episodes already in flight keep running across the swap, so a multi-turn episode can span policy versions; staleness accounts for that. `Dispatcher.on_version_pending` sets `policy_update_pending` and then waits on `scheduling_lock`; `fill_inflight` checks the flag both before and inside the lock; `on_new_version` clears it.
6. **Every trainer DP rank receives the same number of micro batches per step, with the same modality at each index**, so FSDP collectives stay in lockstep. `prepare_batch` (`orchestrator/batch.py`) pads multimodal and text micro batches separately to a multiple of the DP world size, and `balanced_partition` gives each rank an equal count.
7. **The orchestrator stays alive through the trainer's final weight broadcast**, so the trainer is not stranded inside the broadcast handshake. `Orchestrator.wait_for_final_broadcast`, when `max_steps` is set.
8. **Liveness: every wait on another component is bounded or fails when that component dies, and nothing on the weight-swap path waits on `main_loop`.** This is what backs the "never hang silently" goal. `main_loop` and `Orchestrator.wait_for_version` call `_raise_if_component_stopped`, which re-raises a dead dispatcher or watcher task.
9. **Each eval result measures one policy version.**
10. **Emitting a result never blocks; back-pressure acts only on starting new work.** This is what keeps the weight-swap path from waiting on `main_loop`, as the liveness invariant requires. Every emit in `Dispatcher` uses `out_q.put_nowait` on an unbounded queue; `Dispatcher.fill_inflight` pauses all new starts while `is_out_q_backlogged`.

## Known Gaps

Places where today's code does not fully meet an invariant above, or a guarantee a reader would reasonably expect. Fixing a gap means deleting its entry in the same change.

- **The ship hold can wait forever** (invariant: every wait is bounded). `Orchestrator.finalize_train_batch` waits on `version_advanced` with no timeout and without calling `_raise_if_component_stopped`. If the watcher task dies during a hold, `main_loop` hangs silently.
- **A dead trainer goes unnoticed** (every wait is bounded). Within the orchestrator, nothing detects that the trainer has exited: the watcher keeps polling for the next version, the dispatch gate stays closed, and `main_loop` idles without error.
- **Some failures are only logged or lost** (every wait is bounded). `WeightWatcher.apply_policy_update` logs exceptions from `on_version_pending` and `on_new_version` as warnings and continues. `Dispatcher.cancel_inflight` starts its cancellation with `asyncio.create_task` and keeps no reference, so an exception there is never seen.
- **The rate limiter can stall a weight swap** (nothing on the swap path waits). `Dispatcher.acquire` waits on the `tasks_per_minute` limiter while `fill_inflight` holds `scheduling_lock`, and `on_version_pending` waits for that lock, so a swap can wait up to one rate-limit interval.
- **The final-broadcast wait gives up** (stays alive through the final broadcast). `Orchestrator.wait_for_version` stops waiting after `weight_broadcast.timeout` seconds and proceeds to teardown, which can leave the trainer inside the handshake.
- **Wind-down cancellations skip accounting** (every group reaches one terminal state). `Dispatcher.cancel_inflight_train_episodes` (called by `Orchestrator.start_draining` once the step budget is reached) and `Dispatcher.cancel_inflight_episodes` (shutdown) cancel episodes without putting anything on `out_q`. This is harmless today because no train batch ships while draining.
- **A self-cancelled episode is never accounted for** (every group reaches one terminal state). If an episode task ends in a `CancelledError` the dispatcher did not start, `Dispatcher.handle_completed_request` returns without emitting to `out_q`. The group never completes in `TrainSink`, its arrived episodes stay buffered, and its `GroupState` stays in `Dispatcher.groups`.
- **Eval results can mix policy versions** (each eval result measures one version). Eval episodes ignore the weight-swap cancel, so an eval epoch that outlasts a trainer step spans several versions: members started after a swap, and later turns of multi-turn episodes, run on newer weights. `EvalSink` does not read the episodes' policy versions, so the mix goes unflagged.
- **A crashed run still saves a checkpoint.** The `finally` block in `Orchestrator.start` saves an orchestrator checkpoint at the last shipped step even when `main_loop` raised. That step need not have a matching trainer checkpoint.

## Checklist for Changes

Before merging a change to the dispatcher, the orchestrator's main loop or batch shipping, the train sink, concurrency control, the weight watcher, batch transports, or the trainer's data intake:

- Walk the [invariants](#invariants) and name the ones the change could affect. If it changes one on purpose, update this page in the same change.
- If the change fixes a [known gap](#known-gaps), delete the entry; if it finds a new one, add it.
- Put new "start new work?" throttles in `Dispatcher.fill_inflight`, next to the existing ones, not on the output path. Emitting results must never block: the weight-swap path emits while `main_loop` may be waiting on that swap.
- Slow work added to `main_loop` delays all result processing, including group scoring and batch cuts. Offload CPU-heavy work with `asyncio.to_thread`.
- Anything that waits on another component should fail loudly if that component dies, rather than wait forever.
