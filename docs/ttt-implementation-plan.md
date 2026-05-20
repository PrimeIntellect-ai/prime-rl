# Online TTT-LoRA Implementation Plan

This is a forward-looking design doc, written after reading the prior
post-mortem in `docs/test-time-training-lora.md` and a code-archaeology pass
through prime-rl's existing LoRA / multi-run / inference machinery.

The headline finding is that **most of the infrastructure required for
online test-time training already exists in prime-rl** — it was built for a
different purpose (multi-tenant SFT runs sharing one trainer), but the
shape lines up almost 1-to-1 with what TTT needs. The scope of work is
substantially smaller than the prior post-mortem implies; the new code is
mostly session-lifecycle plumbing and a per-rollout learner mode.

The companion document `docs/ttt-cheat-sheet.md` is the one-pager of "do
this / don't do that" pulled from the post-mortem; that's the right
reference for the cluster session.

## Goal recap

Per-rollout LoRA `Phi_ttt` trained online on exact rollout tokens, used to
extend effective context without paying full attention over the long
history every turn. Trainer must replay completion spans under the
adapter snapshot that was active when those tokens were sampled, so
gradients flow into base `Theta` correctly.

Tool-output world-modeling (SFT-on-tool-body) is **independent** and
already landed on `feat/sft-on-tool-outputs`. TTT does not depend on it
and vice versa. As of 2026-05-20 the SFT-on-tool-outputs runs are live
on the cluster and stepping cleanly — the `cmb-both-ct-a0.5` variant is
leading the RL baseline on `forth-lang-test/pass@1`, validating the SFT
path behaviorally.

## Cluster context (2026-05-20)

- H200 (140 GiB) × 8 per node, 2-4 nodes per run.
- `/beegfs` shared storage (140T total, ~25T free).
- Qwen3-4B-Instruct-2507 cached at `/beegfs/huggingface/hub/`.
- vLLM `0.21.0` pinned; weight broadcast type is `nccl` in production.
- Production Forth runs use `optim_cpu_offload=true` to stay under
  140 GiB peak (was 126.6 GiB before); TTT adds learner-side
  forward+backward, so memory accounting matters.
- The trainer-side LoRA defaults are rank=16, alpha=32,
  target_modules covering MLP + attention + MoE variants.
- **Multi-LoRA infrastructure is production-hardened**: same code path
  is used in hosted training where many users each have their own
  adapter, and it's stable. The TTT-specific risk surface is the
  chunked-snapshot churn pattern (32 snapshots/rollout × 512
  rollouts/step), not the multi-LoRA machinery itself.
- **The gate_up_proj layerwise fix landed upstream** (PR #2482,
  `b2ba40b5e` on `main`), so it's in our branches via the ancestor
  chain. The post-mortem's "vLLM crashes on mid-rollout weight update"
  failure mode is no longer a known bug — only a defensive Phase A
  precaution (one less moving part during smoke).
- **Live risk**: per the `daniel/gptoss-lora-nan-repro` HANDOVER, vLLM
  worker state can be poisoned when the same `lora_int_id` is reused
  for a refreshed adapter. The TTT design avoids this by construction:
  each chunk snapshot gets a fresh id and name, never reused. LRU
  eviction handles cleanup. This needs to be a hard invariant, not a
  convention.

## Existing prime-rl machinery we will reuse

These are the load-bearing facts that shape the design.

### Trainer: native multi-LoRA with per-token routing

`src/prime_rl/trainer/lora.py` + `src/prime_rl/trainer/models/layers/lora/`:

- `MultiLoRALinear` wraps `nn.Linear` with `n_adapters` slots and
  `torch._grouped_mm`-based grouped GEMM. A single forward pass can route
  different tokens through different adapter slots.
- Routing is controlled by two globals: `LORA_NUM_TOKENS: [n_adapters]`
  (contiguous token counts per adapter in the flattened batch) and
  `SCALING_FACTORS: [n_adapters]` (per-adapter scale, defaults to a sentinel
  `1e6` so unused slots fail loudly if accidentally used).
- `MultiLoRAGroupedExperts` / `MultiLoRANonGatedGroupedExperts` /
  `MultiLoRAGptOssGroupedExperts` cover the MoE cases.
- `apply_lora_to_model()` is the entry point; it must run **before** FSDP
  setup, so adapter slot count is fixed at trainer boot.

### Trainer: MultiRunManager session lifecycle

`src/prime_rl/trainer/runs.py` + `docs/multi_run_manager.md`:

- Pre-allocates `max_runs` adapter slots in the model.
- Manages bidirectional `run_id <-> idx` mapping, per-run progress,
  per-run optimizer/scheduler via creation/deletion hooks.
- Discovery is filesystem-scan based today (`output_dir/run_*`).
- `evict_run(idx, reason)` is a clean shutdown path; orchestrator picks
  up the reason from `evicted.txt`.
- Already exposes `get_state_dict_for_run(idx)` (HF-compatible adapter
  state dict) and `reset_run_parameters(idx)`.

### Inference: dynamic multi-LoRA via patched vLLM

`src/prime_rl/inference/patches.py` + `src/prime_rl/inference/vllm/server.py`:

- vLLM's native multi-LoRA: `lora_request: LoRARequest(lora_name,
  lora_int_id, lora_path)` per completion request.
- Patched `_patched_load_adapter` and `_patched__apply_adapters` allow
  in-place dynamic loading.
- `POST /load_lora_adapter` endpoint loads an adapter by path; the
  orchestrator scheduler today uses one persistent `lora_name`.

### Orchestrator: continuous batching with per-task state

`src/prime_rl/orchestrator/scheduler.py`:

- `Scheduler` tracks `inflight_requests: dict[asyncio.Task, InflightRequest]`.
- One `lora_name` field today; per-rollout `lora_name` is a small
  extension, not a redesign.

### Verifiers: trajectory_id is already a first-class concept

`deps/verifiers/verifiers/types.py`, `runtime.py`:

- Every rollout has `trajectory_id: str` in its state, set by the
  runtime, propagated into the rollout output.
- This is the natural session identifier — no new id needs to be minted.

### SFT-on-tool-outputs: per-env config pattern

`packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py`:

- `SFTConfig` lives on `TrainEnvConfig.sft`. Per-env opt-in via setting
  `sft.on_tool_outputs = true`.
- `OrchestratorExperimentalConfig` exists but is empty — natural home
  for global `ttt: TTTConfig`.

## Architecture

### Service topology

```
[Orchestrator]
    │
    ├── per-rollout RPC ──▶ [TTT Learner Pool]
    │   (sticky on trajectory_id)    │
    │                                ├── frozen Theta (refreshed on policy update)
    │                                ├── multi-tenant LoRA (n_sessions slots)
    │                                ├── per-session pending-token buffer
    │                                └── chunk snapshots → shared LoRA store
    │                                                          │
    ├── generate request ──▶ [vLLM workers]  ◀─── snapshots ───┘
    │   (lora_name = session_<id>_chunk_<k>)
    │
    └── rollout → [Trainer]
                  │
                  ├── MultiLoRA wrap (replay slots)
                  ├── per-microbatch: lora_num_tokens routes tokens to chunk adapters
                  ├── adapter weights frozen; grads → base only
                  └── reaps consumed snapshots after step
```

### Mapping to existing concepts

| TTT concept           | Existing prime-rl concept                          |
|-----------------------|----------------------------------------------------|
| Session               | "Run" in `MultiRunManager`                         |
| Session id            | `trajectory_id` from verifiers                     |
| Session LoRA          | `MultiLoRALinear` adapter slot                     |
| Chunk snapshot        | Materialized via `get_state_dict_for_run(idx)`     |
| Frozen base           | Trainer model with `freeze_all_except_lora_and_specified` |
| Inference adapter     | `LoRARequest` in vLLM, loaded via patched path     |
| Per-env opt-in        | New flag on `TrainEnvConfig.ttt_enabled`           |
| Global TTT config     | `OrchestratorExperimentalConfig.ttt`               |
| Sticky routing        | Scheduler `inflight_requests` keyed by task        |

This means the bulk of TTT is **wiring existing primitives**, not
inventing new ones.

### TTT learner as a repurposed multi-run trainer

The learner is a separate prime-rl process running in "session-trainer"
mode:

- `setup_multi_run_manager(max_runs = max_concurrent_sessions, ...)`.
- Base model wrapped with `MultiLoRALinear` via `apply_lora_to_model`;
  base is frozen by construction.
- Adapter slots correspond to live sessions; mapping is via RPC, not by
  scanning `run_*` directories.
- Sessions are created (`/start_session`), trained on chunks
  (`/prepare_turn`, `/complete_turn`), snapshotted, and evicted
  (`/finish_session` / `/abort_session`).
- Snapshot path: write `state_dict_for_run(idx)` to
  `adapter_dir/session_<id>/chunk_<k>` (shared FS visible to vLLM and
  trainer).
- Base weight refresh: on `update_weights` from main trainer, swap base
  via the existing NCCL or filesystem broadcast paths; LoRA slots are
  unaffected.

### Chunk-aligned training semantics

Identical to the prior post-mortem; no change. For
`update_every_tokens = N`:

- Tokens `[k·N, (k+1)·N)` are predicted under adapter `A_k`.
- After those tokens arrive, learner trains `A_k -> A_{k+1}` on the
  chunk.
- Snapshot `A_k` is materialized only when needed (lazy: on the next
  generation request that uses it, or when trainer asks for it).

### Replay on the trainer side

This is where the existing MultiLoRA gives us a big simplification.

**Trainer model is wrapped with `MultiLoRALinear` in TTT mode**, with
`n_adapters = R` where `R` is the maximum number of distinct chunk
adapters in any microbatch.

For each microbatch:

1. Look at the microbatch's `ttt_trace` entries. Collect the unique
   `(session_id, chunk_idx)` pairs — call this set `C`.
2. For each `(session, chunk)` in `C`, load its snapshot into a slot.
   The slot mapping is per-microbatch — slots are reused across
   microbatches.
3. Build `lora_num_tokens: [R]` such that token positions belonging to
   chunk `c` go into slot `slot_of(c)`. Tokens not assigned to any
   chunk (e.g., padding or pre-TTT prefix) go into a "no-LoRA" slot
   with `scaling_factor=0` (effectively base-only forward at those
   positions).
4. Forward + backward. LoRA params are **frozen** in this trainer
   (gradients only flow into the base via the FSDP optimizer). The
   adapter contribution still flows through autograd to give the base
   the right effective loss landscape, but the adapter tensors
   themselves don't get updated.
5. Backward step → standard optimizer step on Theta.
6. After all DP ranks finish step, rank 0 reaps consumed snapshots.

**Implication**: the trainer needs an opt-in mode where it loads LoRA
weights as a frozen forward-only contribution. This is a small
extension to the existing `apply_lora_to_model` path — the existing path
sets `requires_grad=True` for LoRA params; we need the opposite for TTT
replay.

The number of forwards per training step is *unchanged* — one forward
per microbatch, just with adapter-routed tokens. This is materially
cheaper than the per-chunk-forward design I sketched before.

### vLLM side

Per-rollout `lora_name` already works in the scheduler; just needs to
be per-task instead of one global string. Per-turn the orchestrator
passes `lora_name = f"sess_{session_id}_chk_{chunk_idx}"` (or some
content hash) into the completion request, and vLLM loads it via the
patched `load_lora_adapter` if not cached.

The patched LRU manager handles cache eviction. We should make
`max_loras` in the vLLM config the binding knob for memory pressure,
sized to roughly `max_concurrent_generation_requests` rather than
`max_concurrent_sessions`.

## Data structures

```python
# transport/types.py — additions to TrainingSample (msgspec.Struct)
class TTTChunk(msgspec.Struct, frozen=True):
    chunk_idx: int
    adapter_handle: str    # path or content hash; opaque to transport
    token_start: int       # offset into the (prompt_ids + completion_ids) joined sequence
    token_end: int
    base_step: int         # Theta version when this snapshot was made

class TrainingSample(...):
    ...existing fields...
    ttt_trace: list[TTTChunk] | None = None
    ttt_session_id: str | None = None
```

The orchestrator builds `ttt_trace` as the rollout proceeds. Packing
preserves it. Microbatch carries a flat representation:

```python
class MicroBatch(...):
    ...
    ttt_chunk_assignment: list[int] | None  # parallel to input_ids; chunk index or -1
    ttt_adapter_handles: list[str] | None   # indexed by chunk index
```

A `-1` chunk assignment means "no LoRA at this position" (handled via
scaling_factor=0 in the dedicated slot).

## Config schema

```python
# packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py

class TTTLoRAConfig(BaseConfig):
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

class TTTConfig(BaseConfig):
    """Global TTT settings. None disables TTT entirely."""
    mode: Literal["online_lora"] = "online_lora"
    window_seq_len: int = 8192
    total_seq_len: int = 32768
    update_every_tokens: int = 1024
    require_exact_token_ids: bool = True
    adapter_dir: Path  # shared FS visible to learner, vLLM, trainer
    lora: TTTLoRAConfig = TTTLoRAConfig()
    learner: TTTLearnerConfig = TTTLearnerConfig()

class TTTLearnerConfig(BaseConfig):
    endpoints: list[str]               # learner addresses; sticky-routed by trajectory_id
    max_sessions_per_learner: int = 64
    session_offload: Literal["cpu_after_request", "gpu_pinned"] = "cpu_after_request"
    unload_vllm_adapters: bool = True
    delete_consumed_adapters: bool = True

class OrchestratorExperimentalConfig(BaseConfig):
    ttt: TTTConfig | None = None       # None = TTT disabled globally

class TrainEnvConfig(EnvConfig):
    ...
    sft: SFTConfig | None = None
    ttt_enabled: bool = False          # per-env opt-in; requires global ttt config
```

**Validation hooks** at orchestrator config load:

1. `experimental.ttt` set → `use_renderer=true` and
   `use_token_client=false`.
2. `experimental.ttt` set → `window_seq_len > max_completion_tokens +
   headroom`. The headroom value should be a small constant (~64 or
   `max_special_tokens_per_turn`) baked into the validator.
3. `experimental.ttt` set → `adapter_dir` writeable from this process.
4. `train.env[i].ttt_enabled = true` → `experimental.ttt` is not None.

## Files that change

Mapped to phases. Phase A is the smoke target.

### Phase A (single learner, one session at a time, disk transport)

- `packages/prime-rl-configs/.../orchestrator.py` — add config classes
  above.
- `src/prime_rl/ttt_learner/__init__.py` — new package
- `src/prime_rl/ttt_learner/server.py` — FastAPI service:
  `/start_session`, `/prepare_turn`, `/complete_turn`,
  `/finish_session`, `/abort_session`, `/update_base_weights`,
  `/health`
- `src/prime_rl/ttt_learner/session.py` — per-session state: pending
  token buffer, chunk boundary tracking, snapshot bookkeeping
- `src/prime_rl/ttt_learner/training.py` — chunk-aligned forward/backward
- `src/prime_rl/orchestrator/ttt_client.py` — async client to learner;
  sticky routing on trajectory_id
- `src/prime_rl/orchestrator/trajectories.py` — hook
  `interleave_rollout` to call learner `prepare_turn`/`complete_turn`
  and accumulate `ttt_trace`; pass `lora_name` per turn
- `src/prime_rl/orchestrator/scheduler.py` — per-task `lora_name` (small
  edit)
- `src/prime_rl/transport/types.py` — `ttt_trace`,
  `ttt_session_id` on `TrainingSample`; `ttt_chunk_assignment`,
  `ttt_adapter_handles` on `MicroBatch`
- `src/prime_rl/trainer/rl/data.py` — propagate ttt fields through
  `_micro_batch_to_tensor`
- `src/prime_rl/trainer/rl/packer.py` — preserve ttt fields through
  packing
- `src/prime_rl/trainer/batch.py` — preserve through `prepare_sample`
- `src/prime_rl/trainer/rl/train.py` — TTT-mode setup: apply LoRA to
  trainer model in frozen mode; per-microbatch load chunk snapshots
  into slots, set `lora_num_tokens` and `scaling_factors`
- `src/prime_rl/trainer/lora.py` — `apply_lora_to_model_frozen` variant
  for replay (or a flag on the existing fn)
- Snapshot reaper: after step commit, rank 0 deletes consumed adapter
  dirs

### Phase B (multi-tenant single learner, per-chunk replay)

- Multi-tenant in learner: per-session state offload to CPU between
  requests
- Per-chunk replay span splitting (currently one-replay-per-rollout)

### Phase C (multi-learner)

- `ttt_client.py`: sticky shard routing
- Admission control (503 on overflow)
- Adapter shm transport instead of disk (`/dev/shm` or vLLM in-process
  adapter API)

### Phase D (perf)

- Multi-tenant batched LoRA training in learner (concatenate sessions'
  chunks into one forward, mask cross-session attention, per-session
  grad accumulation)
- Sliding-window training (cap learner forward at
  `window_seq_len`)
- Replay batching by adapter handle across the global batch

## Hard correctness invariants

These must be enforced at boot or in CI:

1. `experimental.ttt` set → `use_renderer=true, use_token_client=false`.
2. `experimental.ttt` set → `window_seq_len > max_completion_tokens +
   headroom`. Headroom is a small constant; fail loudly otherwise.
3. `vllm.max_lora_rank >= experimental.ttt.lora.rank`.
4. `adapter_dir` writeable from learner, vLLM, trainer (probe at boot).
5. No global tool filter — TTT control surface only exposes
   per-env `ttt_enabled`. Tool-name filtering for SFT lives in
   `TrainEnvConfig.sft.tool_names` (already implemented).
6. `session_id == trajectory_id` — derived, not minted. Survives
   orchestrator restarts and matches rollout traces 1-to-1.
7. **No mid-rollout base weight refresh in Phase A.** Active sessions
   finish under their start-time base; PipelineRL-style updates queue
   instead of preempt. Revisit after smoke.
8. **One TTT sample per rollout in Phase A.** Split spans across
   microbatches only after the span-splitting code is unit-tested
   (Phase B).
9. LoRA replay slots are `requires_grad=False`. Gradient must not flow
   into the adapter. Add a unit test for this.
10. Snapshot deletion is reference-counted: vLLM "done" + trainer "done"
    + DP-step-committed. Never delete on session close alone.
11. **Every chunk snapshot gets a fresh `lora_int_id` and a fresh
    `lora_name`.** Never reuse. Per the
    `daniel/gptoss-lora-nan-repro` HANDOVER, id reuse can poison
    worker-side adapter state. LRU eviction in vLLM handles cleanup.
    Enforce in `ttt_client.py`: id generator is a monotonic counter,
    name encodes `(trajectory_id, chunk_idx)`.

## Performance considerations under high concurrency

In rough priority order:

1. **vLLM adapter delivery via disk vs shm**. Disk is fine for Phase A
   smoke but the volume of snapshots (~32 per rollout at 32k context)
   will get painful. Phase C moves to `/dev/shm` for the hot path; disk
   only as a fallback / for cross-host trainer replay.
2. **Multi-tenant batched LoRA training in learner**. Use the existing
   `MultiLoRALinear` per-token routing to batch multiple sessions'
   chunks in one forward. Phase D.
3. **Replay batching by adapter on trainer**. The MultiLoRA forward
   already routes tokens to slots, so multiple microbatches sharing the
   same chunk-adapter use the same loaded weights without swap. The
   bottleneck is the snapshot load itself; consider mmapping.
4. **Sliding-window training in learner**. Train on the most recent
   `window_seq_len` tokens, not the full session. Bounds per-chunk
   cost. Generation already uses a window, so semantically consistent.
5. **Bounded admission**: learner returns 503 at capacity; orchestrator
   uses existing concurrency knobs to back off.

## Phased delivery

- **Phase A — smoke**: one learner, one session per learner at a time,
  disk transport, one-replay-per-rollout under final adapter, Forth +
  one held-out retrieval task. Goal: correctness end-to-end. The
  one-replay-under-final-adapter simplification is biased for
  multi-chunk completion samples but is fine as a smoke baseline.
- **Phase B — correctness expansion**: multi-tenant per learner with
  CPU offload; per-chunk replay; span-split TTT samples; trajectory-id
  session lifecycle plumbed everywhere; abort-on-failure RPC.
- **Phase C — concurrency**: multi-learner shards with sticky routing;
  admission control; adapter shm transport; vLLM persistent LoRA slot
  rotation.
- **Phase D — perf**: multi-tenant batched LoRA training; sliding-window
  training; replay batching by adapter; async snapshot pipeline.

The prior runs documented in `test-time-training-lora.md` jumped to ~C
and died on items from A. Don't skip A.

## Open questions for the cluster session

Concrete probe specs live in `docs/ttt-probes.md`. Summary of what
each one resolves:

1. **Snapshot churn** — per-adapter vLLM load latency at the pinned
   0.21.0 with Qwen3-4B. Decides disk vs shm transport for Phase A.
2. **Learner forward+backward latency** — 1024-token chunk on a single
   H200. Decides whether single-tenant learner can keep up with
   between-chunk generation gaps.
3. **MultiLoRA forward overhead vs n_adapters** — `_grouped_mm` path at
   `n_adapters ∈ {1, 8, 64, 256, 1024}`. Bounds how wide a replay
   microbatch can be.
4. **`/beegfs` write throughput** — producer-side snapshot write
   latency. Decides whether shm is needed even on the producer.

The SFT smoke task is **already done implicitly** — the live Forth
runs (`forth-lang-qwen-{rl,cmb-code-ct,cmb-both-ct}-r1`) validate the
SFT-on-tool-outputs path. No separate smoke needed.

The dynamic-base-update-with-active-LoRAs failure is **resolved
upstream** (PR #2482) and no longer needs reproduction. We still
queue weight updates between rollouts in Phase A as a conservatism
choice, not because of a known bug.

## Future variants worth pursuing after Phase A works

Two ideas captured here so they don't get lost between now and when
basic chunked TTT validates. Neither changes Phase A scope; both
become natural Phase B+ experiments once the chunked path is real
and we have something to compare against.

### Variant 1: LoRA only on the last N layers

Restrict `target_modules` to a tail slice of the transformer (e.g.,
last 8 of 28 layers) rather than applying LoRA throughout.

**Why it pays off here:**

- Learner-side memory drops in proportion to "how many layers above
  the highest LoRA layer." Activations from earlier layers can be
  discarded after the forward pass; full activation checkpointing
  needs only the top range. Real per-chunk memory win on the learner,
  which is the constraint that bites at high concurrency.
- Backward compute drops too — fewer LoRA layers means fewer
  rank-r matmuls in the backward pass, faster learner step time.
- Configurable via the existing `target_modules` regex; no new code.

**Risk:** whether last-layer-only adapts enough for TTT's actual goal
(compressing trajectory-specific info into weights). The literature
says "fine for most fine-tuning tasks, worse for ones needing deep
representation shifts." For TTT the goal is closer to
memorization-of-a-specific-trajectory than learning-a-new-task, so
last-layer-only is probably enough — but it's empirical, ablate
side-by-side.

**Status:** add as a config flag, default to all-layer LoRA in Phase A,
run an ablation early in Phase B.

### Variant 2: Compaction-aligned TTT with summary-as-bridge

Instead of (or in addition to) updating every `update_every_tokens`,
fire a TTT update at the moment a long-context harness compacts the
trajectory — i.e., right before the model summarizes and dismisses
the prior conversation. The training data for the update is the
pre-compaction context plus the summary itself.

**The summary-as-bridge framing.**

The summary plays a double role:

1. It's the token sequence the model emits *before* compaction, so
   it's a natural training target conditioned on the pre-compaction
   context.
2. It's the token sequence the model attends to *after* compaction.

Training the LoRA on `(pre_compaction_context → summary)` makes the
summary act as a learned **key** into the LoRA's encoded
representation of the dismissed context. Post-compaction, the model's
attention over the summary token activates the LoRA's stored
knowledge. The summary is the bridge — in a literal sense, it's the
only token sequence that exists in both pre- and post-compaction
attention contexts. Build a key-value memory where the keys are the
model's own summaries, with no separate retrieval mechanism, no extra
parameters, no new losses.

**Why this is interesting compared to chunked TTT:**

| | Chunked TTT | Compaction-aligned |
|---|---|---|
| Updates per rollout (32k context) | ~32 | 1–3 |
| Snapshot churn at steady state | ~16k live | a few hundred |
| Replay complexity | per-chunk forwards | per-compaction forwards |
| vLLM adapter cache pressure | high | low |
| Storage / reaper pressure | the main bottleneck | trivial |
| Signal density per update | low (random chunk boundaries) | high (semantic boundaries) |

A bunch of the perf worries in this plan (snapshot transport,
adapter-dir scaling, reaper aggressiveness) become much smaller
problems if compaction-aligned ends up as the primary mode.

**Property worth noting:** because compaction triggers on a configurable
token threshold (`summarize_at_tokens` in RLM today), either the
rollout is short enough that TTT wouldn't help anyway *or* the model
compacts and TTT fires at exactly the right moment. The control
surface is simple.

**Detecting compaction events from the trajectory.**

The RLM harness today doesn't emit a typed compaction event, but the
detection signal is purely structural — no string matching required.
From `deps/verifiers/.../harnesses/rlm.py:33–66`:

A trajectory step whose `prompt` is *shorter* than the accumulated
conversation up to that point means the harness reset the conversation
to `[system, user(framing + summary)]`. That length comparison is the
hard signal:

```python
def detect_compaction_events(trajectory):
    if not trajectory:
        return
    prev_len = len(trajectory[0]["prompt"]) + len(trajectory[0]["completion"])
    for i, step in enumerate(trajectory[1:], start=1):
        if len(step["prompt"]) < prev_len:
            yield i  # compaction event at this step boundary
        prev_len = len(step["prompt"]) + len(step["completion"])
```

The `COMPACTION_BOUNDARY_MARKER` string in the harness is a debug aid
for human-readable rendering; we don't depend on it.

**Sub-LLM handling is already in place.**

The RLM harness sets `X-RLM-Depth: 0` on parent calls and increments
for each sub-agent level. The existing `keep_trajectory_step` filter
in `harness.py` elides depth>0 calls before they reach prime-rl, so
the trajectory consumed for training already contains only
parent-agent steps. Compaction signals apply to parent-level
compaction events, which is exactly what we want. Sub-agents have
their own internal context but their compactions don't surface in the
parent-agent trajectory — that's correct behavior, since we wouldn't
be training on sub-agent contexts anyway.

**V1 RLM: propose typed compaction metadata.**

The structural inference works today but is a "we know what shape it
has" trick. When the v1 RLM lands its compaction logic, the right
move is to make compaction a typed event in the trajectory rather
than something inferred. Two natural shapes:

*Option A: per-step metadata (preferred).*

```python
class TrajectoryStep(TypedDict):
    prompt: Messages
    completion: Messages
    ...
    metadata: NotRequired[StepMetadata]

class StepMetadata(TypedDict, total=False):
    compaction: CompactionInfo | None

class CompactionInfo(TypedDict):
    dismissed_message_range: tuple[int, int]   # [start, end) into pre-compaction messages
    summary_message_index: int                  # which message in step["prompt"] is the summary
    pre_compaction_token_count: int             # sanity / debugging
```

*Option B: a top-level events list on `State`.*

```python
class State(TypedDict):
    ...
    events: NotRequired[list[TrajectoryEvent]]
```

Prefer A: compaction is inherently step-bound, so co-locating the
metadata with the step that triggered it keeps the data cohesive.
Use B only if/when a second event type appears that doesn't naturally
bind to a step.

The harness has all the info needed to populate `CompactionInfo`
(it just dismissed those messages and produced that summary).
Emitting metadata is ~5 lines on the harness side, zero work for the
consumer beyond reading the field.

**Migration path — three tiers, no coordinated cutover.**

1. **Today.** Structural inference from `state["trajectory"]` step
   prompt lengths. ~20-line parser in prime-rl. Works on the existing
   RLM harness without any verifiers-side change.
2. **Soon.** Typed `CompactionInfo` metadata on trajectory steps,
   emitted by the v1 RLM harness. Cleaner contract, ~5 LOC in the
   harness. Prime-rl parser adds an "if metadata present, use it; else
   fall back to structural" branch.
3. **Eventually.** Generalize metadata into `step.metadata.events`
   if/when a second harness needs typed signals of a different kind.
   Skip until there's a real second use case.

**Constraint.** Compaction is RLM-harness-specific today. Other envs
(`opencode`, `mini_swe_agent`, anything outside the experimental
composable framework) don't compact, so compaction-aligned TTT is
RLM-only in the short term. Chunked TTT remains the universal
fallback for non-RLM rollouts.

**Status.** Phase A stays chunked-only (universal coverage, simpler
to validate end-to-end). Phase B adds compaction-aligned as an
alternate `update_strategy`, reusing the chunk-train API with
compaction-event boundaries. If the comparison experiment favors
compaction-aligned + bridge training (likely for RLM specifically),
it becomes the default for RLM and chunked stays as the fallback.

## Minimal safe run checklist

Pulled from the post-mortem; the cheat-sheet has the full version. The
TTT-specific ones:

1. Branches: code on `feat/ttt-online-lora`, configs on the matching
   branch in research-configs.
2. `orchestrator.use_renderer = true`, `use_token_client = false`.
3. `window_seq_len > max_completion_tokens + headroom`.
4. `max_total_completion_tokens` carries the long-trajectory budget.
5. vLLM `enable_lora=true`, `max_lora_rank >= ttt.lora.rank`,
   `max_loras` sized to concurrent generation requests.
6. `adapter_dir` on shared FS visible to learner + vLLM + trainer.
7. Learner concurrency sized for the load — start small, profile.
8. SFT tool-name filters per-env, never global.
9. Eval envs don't inherit train-only filters.
10. Sessions close after train and eval rollouts.
11. Logs show real prompt budgets, not `prompt_tokens=1`.
12. W&B shows nonzero rollout progress and sane eval completion rates
    before treating the run as valid.
