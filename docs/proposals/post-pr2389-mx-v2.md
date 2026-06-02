# RFC: `weight_broadcast.type = "mx_v2"` — the complete prime-rl × ModelExpress design

> **Status:** Draft. Targets [`PrimeIntellect-ai/prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) upstream as a follow-up to [PR #2389](https://github.com/PrimeIntellect-ai/prime-rl/pull/2389).
>
> **Companion docs in this branch:**
> - [`post-pr2389-kernel-compile-plan.md`](./post-pr2389-kernel-compile-plan.md) — the four-phase plan this RFC consolidates
> - [`post-pr2389-status-and-plan.md`](./post-pr2389-status-and-plan.md) — current status of the four phases
> - [`build-notes-2026-05-28.md`](./build-notes-2026-05-28.md) — image-build mechanics for the source-baked Phase 2 + Phase 3 image
> - [`image-build-mx-v2.md`](./image-build-mx-v2.md) — the v2 overlay-image plan (Workstream B of this RFC)

## TL;DR

This RFC proposes a single new weight-broadcast type, **`weight_broadcast.type = "mx_v2"`**, that contains every optimization the post-#2389 RFC identifies, behind one config knob. It coexists with the existing `"nixl_mx"` (PR #2389) for migration; no behavior of `"nixl_mx"` changes.

| Capability | `nixl_mx` (PR #2389) | `mx_v2` (this RFC) |
|---|---|---|
| Data plane | NIXL RDMA WRITE (push) | NIXL RDMA WRITE *or* READ (engine-dispatched) |
| Control plane | In-tree `MxRendezvous` (185 LOC, 4 gRPC calls) | `MxWeightTransferEngine` over MX v2 fat clients |
| Heartbeat + freshest-dedup + same-rank routing | Runtime monkey-patch via configmap | Baked in (Phase 2) |
| compile_target safety net | None — silent corruption on layout mismatch | Phase 3 — refuses mismatched at discovery, before RDMA |
| Mixed-TP / mixed-EP | Requires matching layouts | Phase 4 — multi-source slice picker + stitching |
| Tree fan-out (pipeline replication) | None — single-source from trainer | Receivers republish; trainer NIC stops being the bottleneck past ~4 receivers |
| MoE expert filter | None — every receiver pulls every expert | Bandwidth-proportional to EP (8× savings for EP=8 on a 192-expert model) |
| vLLM native API alignment | Bespoke worker extension | Targets `WeightTransferEngine` ABC (same shape as Anyscale RDT PR #43375) |
| Net LOC | baseline | +255 (new) / −185 (delete in-tree rendezvous in follow-up) |

## Motivation

PR #2389 ships a working NIXL+MX path on GB200 (~10 s/cycle on Qwen3-30B-A3B). Production hardening since then surfaced six issues that map to one root cause: **prime-rl owns the rendezvous + transport layer, so every cross-cutting capability (heartbeat, compile-target metadata, slice picking) lives in prime-rl too.**

The cross-cutting work — heartbeat, dedup, shape registry, MoE expert filter, mixed-TP slice picker, tree fan-out — has been built and unit-tested **inside ModelExpress** (PR [#349](https://github.com/ai-dynamo/modelexpress/pull/349), branch `kavink/post-2389-phase3-4`). What's missing is the single integration PR that consumes all of it from prime-rl.

This RFC is that integration PR.

## Design

### 1. New config type

`packages/prime-rl-configs/src/prime_rl/configs/trainer.py` adds `MxV2WeightBroadcastConfig`:

```python
class MxV2WeightBroadcastConfig(BaseWeightBroadcastConfig):
    type: Literal["mx_v2"] = "mx_v2"

    # ─── Control plane ──────────────────────────────────────────────
    host: str = "localhost"
    port: int = 29501
    timeout: int = 1200

    # ─── Discovery (Phase 2) ────────────────────────────────────────
    same_rank_only: bool = True
    """GB200/EFA multi-NIC fabrics: receivers pull from same-rank trainer only."""
    dedup_freshest_per_rank: bool = True
    """When multiple READY entries share a worker_rank, pick the freshest by updated_at."""

    # ─── Layout metadata (Phase 3) ──────────────────────────────────
    publish_compile_target: bool = True
    """Trainer stamps every publish with the conversion's compile_target tag."""
    compile_target_filter: list[str] | None = None
    """Receiver-side whitelist. None = accept anything (back-compat). Set to
    {'cutlass_fp8'} or {'hf_raw','cutlass_fp8'} to refuse mismatches before RDMA."""

    # ─── Sharding (Phase 4) ─────────────────────────────────────────
    target_tp_layout: TargetTPLayout | None = None
    """None = matched-TP fast path (single-source same-rank pull).
    Set when trainer TP/EP layout differs from inference."""

    # ─── Pipeline replication (TensorHub pattern) ───────────────────
    publish_self_as_replica: bool = True
    """After a successful receive, inference workers republish themselves as
    sources; subsequent receivers can pull from peers instead of the trainer."""

    inference_world_size: int = 1
    inference_model_name: str = ""
```

### 2. Selector dispatch

`src/prime_rl/trainer/rl/broadcast/__init__.py` adds one `elif`:

```python
elif config.type == "mx_v2":
    from prime_rl.trainer.rl.broadcast.nixl_mx_v2 import NIXLMxV2WeightBroadcast

    assert parallel_dims is not None, "mx_v2 requires parallel_dims"
    return NIXLMxV2WeightBroadcast(output_dir, config, parallel_dims)
```

### 3. New trainer broadcast — `src/prime_rl/trainer/rl/broadcast/nixl_mx_v2.py`

Replaces the bespoke `MxRendezvous` + manual NIXL `post_write` flow with `MxV2TrainingPublisher`:

```python
class NIXLMxV2WeightBroadcast(WeightBroadcast):
    def __init__(self, output_dir, config, parallel_dims):
        self.publisher = MxV2TrainingPublisher(
            agent_name=make_agent_name("trainer", world.rank),
            device_id=torch.cuda.current_device(),
            mx_server_url=f"{config.host}:{config.port}",
            worker_rank=world.rank,
            world_layout=TrainerWorldLayout(...),  # from parallel_dims
        )

    def lazy_init(self, model):
        self.publisher.initialize(model_name=config.inference_model_name)
        # Slot allocation + conversion still owned by prime-rl
        self.model_slots = model.build_slots(...)
        self.conversion = select_default_conversion(...)

    @torch.no_grad()
    def broadcast_weights(self, model, step):
        # 1. Run trainer-side conversion (prime-rl owns this)
        for slot in self.model_slots:
            slot.fill_from(model.state_dict(), self.conversion)

        # 2. Register each slot tensor with the publisher, tagged with
        #    compile_target + compile_metadata from the conversion registry
        for slot in self.model_slots:
            for name, tensor, _ in slot.buffers:
                self.publisher.add_tensor(
                    name=name,
                    tensor=tensor,
                    compile_target=self.conversion.compile_target,
                    compile_metadata=self.conversion.compile_metadata,
                    # MoE expert metadata where applicable
                    is_expert=slot.is_expert,
                    expert_axis=slot.expert_axis,
                    owned_expert_ids=slot.owned_expert_ids,
                )

        # 3. One publish() per step
        self.publisher.publish(version=step)
        self.publisher.mark_ready()
```

**Key invariants preserved from PR #2389:**
- Trainer-side conversion (FP8 packing, fusion, sharding) — prime-rl owns the kernel
- Slot layout — `Sharded` / `Gathered` / `Expert` slots stay
- HSDP barrier — `dp_replicate > 1` only publishes from rank 0
- Per-step lifecycle — `lazy_init` on first call, `broadcast_weights` every step

**What changes vs PR #2389:**
- The push (`nixl_agent.post_write` loop) becomes a publish (`publisher.add_tensor` × N + `publisher.publish`); the actual NIXL WRITE is now driven from the receiver side via `receive_weights_scratch`
- Trainer no longer needs to know inference world size in advance — receivers discover via catalog

### 4. New inference worker — `src/prime_rl/inference/vllm/worker/nixl_mx_v2.py`

Uses `MxWeightTransferEngine` via vLLM's `worker_extension_cls`:

```python
class NIXLMxV2WeightUpdateWorker(Worker):
    """vLLM worker extension for the v2 pull path."""

    def init_nixl_mx_v2(self, host: str, port: int, rank_offset: int, **engine_init_kwargs):
        from modelexpress.vllm_weight_transfer import MxInitInfo, MxWeightTransferEngine
        global_rank = rank_offset + self.device.index
        inference_model_name = self.model_runner.model_config.model

        self.engine = MxWeightTransferEngine(init_info=MxInitInfo(
            mx_server_url=f"{host}:{port}",
            model_name=inference_model_name,
            worker_rank=global_rank,
            agent_name=make_agent_name("inference", global_rank),
            device_id=self.device.index,
            publish_self_as_replica=engine_init_kwargs.get("tree_fanout", True),
        ))

    @torch.no_grad()
    def update_weights_via_mx_v2(self, step: int, *, compile_target_filter=None, target_tp_layout=None) -> None:
        from modelexpress.vllm_weight_transfer import MxUpdateInfo
        self.engine.receive_weights(
            MxUpdateInfo(
                version=step,
                compile_target_filter=set(compile_target_filter) if compile_target_filter else None,
                target_tp_layout=target_tp_layout,
                timeout_seconds=self.config.timeout,
            ),
            load_weights=self._load_weights_batch,
        )
        # Same post-load housekeeping as PR #2389
        update_mla_absorbed_weights(self.raw_model)

    def _load_weights_batch(self, batch: list[tuple[str, torch.Tensor]]) -> None:
        """Feed yielded tensors through vLLM's model.load_weights().
        vLLM handles HF→fused name remapping via stacked_params_mapping."""
        self.raw_model.load_weights(batch)
```

**Key changes vs PR #2389:**
- No pre-registered NIXL buffers on inference side (uses `receive_weights_scratch` under the hood)
- Trainer push → receiver pull semantics
- HF-format publish → vLLM `load_weights` handles fused param remapping (matches NeMo-RL pattern)

### 5. Conversion registry — already done

`src/prime_rl/trainer/models/conversions/__init__.py` was extended on `kavink/post-2389-conversion-registry-extensions` ([Draft PR #2](https://github.com/KavinKrishnan/prime-rl/pull/2)) with:
- `compile_target` + `compile_metadata` fields on `ConversionEntry`
- `cutlass_fp8_e4m3_per_channel` registered alongside `bf16_cast` and `fp8_128x128`
- 19/19 unit tests green

This RFC just consumes that work — no additional changes to conversion-registry code.

### 6. Image — overlay on `v0.7.1-kavin-phase2-phase3`

The v2 source layers cleanly on top of `prime-rl-mx-on-nixl:v0.7.1-kavin-phase2-phase3`, which already contains Phase 2 + Phase 3. The Dockerfile is a 5-line overlay:

```dockerfile
FROM nvcr.io/nvidian/dynamo-dev/prime-rl-mx-on-nixl:v0.7.1-kavin-phase2-phase3
COPY src/prime_rl/transport/                          /app/src/prime_rl/transport/
COPY src/prime_rl/inference/vllm/worker/nixl_mx_v2.py /app/src/prime_rl/inference/vllm/worker/nixl_mx_v2.py
COPY src/prime_rl/trainer/rl/broadcast/nixl_mx_v2.py  /app/src/prime_rl/trainer/rl/broadcast/nixl_mx_v2.py
COPY src/prime_rl/trainer/rl/broadcast/__init__.py    /app/src/prime_rl/trainer/rl/broadcast/__init__.py
COPY packages/prime-rl-configs/src/prime_rl/configs/trainer.py /app/packages/prime-rl-configs/src/prime_rl/configs/trainer.py
```

Plus a 1-line `uv sync --no-deps --reinstall-package modelexpress` if we need to pull in the MX-side engine adapter that ships with PR #349.

See [`image-build-mx-v2.md`](./image-build-mx-v2.md) for the build mechanics + cluster deployment steps.

## Migration

| Phase | Config | Status |
|---|---|---|
| **v0.x** (now) | `nixl_mx` and `mx_v2` coexist | `nixl_mx` remains the documented default. `mx_v2` opt-in. |
| **v0.x+1** | `nixl_mx` deprecated with warning | After 4 weeks of `mx_v2` bake-time on `kavin` + at least one external user. |
| **v0.x+2** | `nixl_mx` removed | After another release cycle. Tracks vLLM's native `WeightTransferEngine` API merge — once that's available, `mx_v2` registers as `backend="mx_nixl"` and `WeightTransferConfig` becomes the recommended entry point. |

No user is forced to migrate. `nixl_mx` users get heartbeat + dedup via PR #1 (Phase 2 source-bake) regardless of this RFC.

## Validation plan

### Pre-merge (this branch, before opening upstream)

1. **Unit tests:** existing 58 MX-side tests (35 v2 shape/picker + 14 engine + 9 bench) + new tests for the prime-rl integration files (≥10 new unit tests covering `NIXLMxV2WeightBroadcast.broadcast_weights` and `NIXLMxV2WeightUpdateWorker.update_weights_via_mx_v2`).

2. **Cluster A/B on `kavin` namespace, GB200, Qwen3-30B-A3B-Instruct-2507:**

   | Config | Refit cycle | Bandwidth | Notes |
   |---|---|---|---|
   | `type=nixl_mx` (PR #2389 baseline) | target ~10 s | ~80 ms NIXL push | Push, no filter, no fan-out |
   | `type=mx_v2`, defaults | target ≤ 10 s | should match | Pull, filter on, fan-out on but with 1 inference replica (no-op) |
   | `type=mx_v2` + 4 inference replicas | target ≤ 10 s | `fanout_factor > 1.0` | Tree fan-out kicks in |
   | `type=mx_v2` + mismatched filter | should refuse at discovery | 0 RDMA bytes | Compile-target safety net under production workload |

3. **Elastic scale-up:** scale inference from 2 → 4 replicas mid-training. With Phase 2 same-rank routing baked in, all 4 should join cleanly without orchestrator restart. Measured via the harness in `modelexpress/benchmarks/bench_elastic_scaling.py` but against the real model, not synthetic tensors.

### Post-merge (upstream)

1. Add `mx_v2` smoke test to upstream prime-rl CI.
2. Coordinate with upstream PrimeIntellect on a real-RL-job validation matrix.

## What this RFC does *not* do

| Out of scope | Why |
|---|---|
| Delete PR #2389's `nixl_mx` code | Coexist for ≥1 release cycle. PR #1 (Phase 2 fixes) lands into `nixl_mx` regardless. |
| Implement delta-sync ([HF blog](https://huggingface.co/blog/delta-weight-sync)) | Layer-2 optimization — composes orthogonally with `mx_v2` and lands separately once vLLM merges `pause_generation(mode="keep")`. |
| Implement true async refit (Composer 2 / Fireworks) | Layer-2 optimization. Same reason as delta-sync. |
| Cross-DC / WAN | TensorHub-pattern; `mx_v2` already supports it via MX catalog metadata, but no cross-DC validation here. |
| Production hardening of MX server | Owned by `ai-dynamo/modelexpress`. We consume; we don't fork. |

## References

- [PR #2389](https://github.com/PrimeIntellect-ai/prime-rl/pull/2389) — the baseline
- [`KavinKrishnan/prime-rl#1`](https://github.com/KavinKrishnan/prime-rl/pull/1) — Phase 2 source-baked rendezvous fixes (heartbeat + dedup + same-rank)
- [`KavinKrishnan/prime-rl#2`](https://github.com/KavinKrishnan/prime-rl/pull/2) — Phase 3 conversion-registry extensions (compile_target + cutlass_fp8)
- [`ai-dynamo/modelexpress#349`](https://github.com/ai-dynamo/modelexpress/pull/349) — Phase 3 + 4 + `MxWeightTransferEngine` adapter (v2 fat clients, multi-source slice picker, vLLM API adapter)
- [vLLM PR #43375](https://github.com/vllm-project/vllm/pull/43375) — Anyscale Ray Direct Transport; same `WeightTransferEngine` API shape, complementary Ray-based catalog choice
- [vLLM native RL APIs blog](https://blog.vllm.ai/2026/05/28/native-rl-apis.html) — the upstream API surface this RFC targets
- [TensorHub paper (arXiv 2604.09107)](https://arxiv.org/pdf/2604.09107v1) — Reference-Oriented Storage, pipeline replication, mutability contract
- [`post-pr2389-kernel-compile-plan.md`](./post-pr2389-kernel-compile-plan.md) — the four-phase plan this RFC consolidates
