# Post-PR-#2389 plan: kernel-compile separation, mixed-TP, and MX client adoption

**Status**: Planning doc. Branch `kavink/post-2389-kernel-compile-plan` (NVIDIA-authored, off PR [#2389](https://github.com/PrimeIntellect-ai/prime-rl/pull/2389) HEAD `79ea824d8`).
**Premise**: This plan is what we propose to build on top of `nixl_mx` once #2389 merges to `main`. It (a) graduates the in-tree `MxRendezvous` reimplementation onto NVIDIA's published ModelExpress clients, (b) introduces a compile-target registry to fix the trainer-side cutlass-pinning issue surfaced during #2389's FP8 cast-pipeline iteration, and (c) extends the v2 shape registry to handle mixed-TP / sharded-source transfers. **None of this fights the #2389 data plane** — the `Slot` / `TransportPlan` / `NixlAgentWrapper` / `classic_cuda_pool` stack stays untouched. We extend the rendezvous and metadata surfaces only.

---

## 1. What we're building on

After #2389 merges, prime-rl's `nixl_mx` mode has:

- `src/prime_rl/transport/` — `NixlAgentWrapper`, `MxRendezvous` (PI's reimplementation of MX's upper layer), `TransportPlan`, `classic_cuda_pool`, `wire.py` (msgspec types).
- `src/prime_rl/trainer/models/slots.py` — `ShardedSlot`, `GatheredSlot`, `ExpertSlot` (the per-tensor buffer abstractions that hold registered NIXL memory and conversion logic).
- `src/prime_rl/trainer/models/conversions/` — bf16-cast and FP8-blockwise conversion specs (`bf16_cast.py`, `fp8_blockwise.py`).
- `src/prime_rl/trainer/rl/broadcast/nixl_mx.py` — `NIXLMxWeightBroadcast`, the lifecycle wrapper.
- `src/prime_rl/inference/vllm/worker/nixl_mx.py` — `NIXLMxWeightUpdateWorker`, the vLLM worker extension.

PI's data plane (Slot / TransportPlan / NixlAgentWrapper / FP8 conversion specs) is **kept**. It's the conversion / compile / topology layer where this plan extends things.

---

## 2. The problem Matej flagged: trainer-side kernel compile pins the topology

In #2389 today, the conversion-cast pipeline (`fp8_blockwise.py`, `Slot.convert`, the recent `c3c4b148` "inline transfer slot casting" + `47e170f5` "simplify transfer cast conversions" refactors) runs **on the trainer**. The trainer's send bucket holds the **post-compile** layout: bf16 if the inference engine wants bf16, FP8 with DeepGemm scale interleaving if it wants DeepGemm, cutlass-friendly column-major + epilogue scales if it wants cutlass, etc. Inference RDMA-reads the bucket and copies straight into live params.

This is fast — no receiver-side compute on the refit critical path — but it couples three independent decisions into the trainer's process:

1. **Which kernel format does inference want?** Today the trainer has to know. Adding cutlass on GB300 means trainer code changes; adding DeepGemm-EP changes the slot layout.
2. **Mixed inference replicas with different kernels.** With trainer-side compile, one trainer can serve only one compile target per refit. Heterogeneous fleets (e.g. a/b testing DeepGemm vs cutlass on the same training run) need two parallel trainer buckets.
3. **Mixed TP/EP layouts.** A trainer at TP=4 publishing to inference at TP=8 needs to know the inference splits at *publish* time to write the right bucket entries. The information is in PI's `build_topology`, but it's tangled with the compile decision because the compile passes produce TP-specific layouts.

When Matej said "we're having issues handling compiled kernels" — this is the source. The cutlass compile happens trainer-side; any inference replica that doesn't want exactly the trainer's compile target either gets bad bytes or fails the assertion check.

---

## 3. Proposed shift: receiver-side compile via scratch buffers

Move the compile pass back to the **inference side**. Trainer ships **canonical HF layout** over NIXL (raw bfloat16 or raw float8_e4m3fn + raw per-block scales — whatever format the trainer naturally has post-optimizer-step). Inference runs the kernel-specific compile pass after the RDMA receive completes, into its own live params.

This is the same scratch-buffer pattern we already validated in two places:

- Our **original PrimeRL PoC** (`KavinKrishnan/prime-rl:kavink/mx-weight-broadcast`) used scratch buffers explicitly to triangulate KL drift — proved correctness when the receiver decides the layout.
- **John Thompson's NemoRL + Dynamo path** (May 22, 2026) uses `MxRefitReceiver.receive_weights_scratch` to handle vLLM's HF→fused param remapping via `stacked_params_mapping`. Validated end-to-end on GB300 RoCE at **380 Gbps** for an 8.82 GB / 399-tensor refit — same scratch path we'd reuse here, just with kernel-compile transforms instead of name remapping.

### Receiver-side refit pseudocode

```python
# inference/vllm/worker/nixl_mx.py — after PR #2389 graduates to MX clients

def update_weights_via_mx(self, *, version, mx_config):
    # 1. Discover same-rank trainer source via MxV2RefitReceiver
    candidates = self._mx_receiver.discover_v2_sources(
        model_name=self.model_name,
        min_version=version,
        same_rank_only=mx_config.same_rank_only,
        compile_target_filter=self._target_kernel,  # NEW — see §5
    )
    chosen = self._mx_receiver.pick_best_source(candidates)

    # 2. RDMA pull into scratch buffers (HF layout — whatever trainer published)
    scratch = {}
    for name, tensor in self._mx_receiver.receive_weights_scratch(
        chosen.ref,
        tensor_shapes=chosen.registry.tensor_shapes,   # global shapes from v2 sidecar
        target_tp_layout=self._tp_layout,              # NEW — mixed-TP slice request
    ):
        scratch[name] = tensor

    # 3. Run the kernel-specific compile pass into live model params
    self._compile_pass.apply(
        scratch_buffers=scratch,
        live_params=dict(self.model_runner.model.named_parameters()),
    )

    # 4. Tree fan-out republish (TensorHub pipeline replication) — unchanged
    self._mx_receiver.publish_self_as_source(...)
```

The compile pass is a small dispatch:

```python
# inference/vllm/worker/compile_passes.py — NEW

class CompilePass(ABC):
    target: str
    def apply(self, scratch_buffers, live_params): ...

class HFRaw(CompilePass):
    target = "hf_raw"
    def apply(self, scratch, live):
        # Direct copy — names + dtypes match. fast path.
        for name, t in scratch.items():
            live[name].data.copy_(t)

class DeepGemmFP8(CompilePass):
    target = "deep_gemm_fp8"
    def apply(self, scratch, live):
        # K-major scale interleave, fused gate_up_proj packing
        for name, t in scratch.items():
            interleaved = deep_gemm_layout(t, block_size=128)
            live[fused_name(name)].data.copy_(interleaved)

class CutlassFP8(CompilePass):
    target = "cutlass_fp8"
    def apply(self, scratch, live):
        # Column-major weights + epilogue scale tensors
        ...
```

### Cost vs benefit

**Cost**: extra GPU memory for scratch (~1× model size briefly per refit, freed after compile) + compile latency on the inference side (~50-200ms for FP8 passes on Qwen3-30B-A3B).

**Benefit**:
- Trainer is **kernel-agnostic** — same bucket bytes serve any inference target.
- Mixed-fleet OK — different inference replicas can run different compile passes on the same trainer publish.
- Adding a new kernel = adding a new `CompilePass` subclass on the inference side. **Zero trainer change.**
- Cross-TP / cross-EP layouts decouple cleanly from compile (see §6).

This is the same trade John Thompson made for Dynamo and the same trade we made in the original PoC. Empirically the 50-200ms compile latency is dwarfed by the 200ms RDMA pull anyway; total wall time is unchanged.

---

## 4. New primitive: compile-target registry (extension to v2 shape registry)

Our v2 shape registry today encodes:

```
TensorDescriptorV2:
  name, global_shape, dtype, placement_kind, shard_axis, local_shard_range,
  is_expert, expert_axis, owned_expert_ids
```

Add **two fields** at the registry level (not per-tensor — these describe how the *publisher* prepared the data):

```diff
 RegistryPayload (JSON in __mx_v2_meta__ sidecar):
   version: int
   trainer_world_layout: str  # "fsdp:4,tp:1,pp:1,ep:1"
+  compile_target: str         # "hf_raw" | "deep_gemm_fp8" | "cutlass_fp8" | ...
+  compile_metadata: dict      # kernel-specific params:
+                              #   {block_size: 128, scale_dtype: "float8_e8m0",
+                              #    layout: "row_major", ...}
   tensors: list[TensorDescriptorV2]
```

### Trainer side

The trainer declares what it published — typically `"hf_raw"` post-optimizer-step:

```python
publisher = MxV2TrainingPublisher(
    agent_name=...,
    world_layout=TrainerWorldLayout(fsdp_world_size=4, tp_world_size=1),
    compile_target="hf_raw",   # NEW — declarative, no inference-side dependency
)
```

Specialized trainers that *do* want to bake a compile pass in for perf can declare it:

```python
publisher = MxV2TrainingPublisher(
    ...,
    compile_target="deep_gemm_fp8",
    compile_metadata={"block_size": 128, "scale_dtype": "float8_e8m0"},
)
```

— and then only same-target inference replicas will accept that publish. Mixed-target inference replicas skip it and look for an `hf_raw` source.

### Receiver side

`discover_v2_sources(compile_target_filter=...)` filters candidate trainers:

```python
candidates = receiver.discover_v2_sources(
    model_name=...,
    min_version=N,
    same_rank_only=True,
    compile_target_filter={"hf_raw"},    # accept HF-raw only, run compile myself
)
```

Or accept any compatible target (if the receiver has a fallback compile pass):

```python
candidates = receiver.discover_v2_sources(
    model_name=...,
    compile_target_filter={"hf_raw", "deep_gemm_fp8"},  # accept either
)
```

The picker's existing trainer-vs-replica + freshest-per-rank sort applies on the filtered set.

### Why this is in the v2 shape registry, not a new transport

It's metadata about the wire format. The shape registry already travels via the synthetic `__mx_v2_meta__` `TensorDescriptor` sidecar (proven by jthomson04's GB300 run + protected by PR #295's filter). Adding two more JSON fields is zero-cost on the wire and keeps the discovery contract in one place.

---

## 5. Sharding and mixed-TP transfers

The hardest case is **trainer TP/EP layout ≠ inference TP/EP layout**. The shape registry already encodes this on the publish side:

```
placement_kind: "SHARD"
shard_axis: 0
local_shard_range: (start, end)   # this rank's slice along shard_axis
```

What's missing is the **receiver's expression of what it wants**.

### New receiver-side API

```python
class TargetTPLayout:
    """What slice of the global tensor THIS receiver needs."""
    world_size: int       # inference TP world size
    rank: int             # this receiver's TP rank
    shard_axis: int       # which axis we're sharded on (model-dependent)

receiver.receive_weights_scratch(
    chosen.ref,
    target_tp_layout=TargetTPLayout(world_size=8, rank=3, shard_axis=0),
    tensor_shapes=chosen.registry.tensor_shapes,
)
```

The receiver computes its desired slice from `target_tp_layout` and the published `placement_kind`:

| Publisher | Receiver | Result |
|---|---|---|
| `REPLICATE` (trainer TP=1) | TP=8, rank=3, axis=0 | Receiver requests rows `[3N/8 : 4N/8]` of the publisher's tensor |
| `SHARD(0)` trainer TP=4 rank=2, range `[N/2 : 3N/4]` | TP=8, rank=4, axis=0, requests `[N/2 : 5N/8]` | Receiver pulls from same-physical-rank trainer (R2), takes lower half of R2's shard |
| `SHARD(0)` trainer TP=4 rank=2, range `[N/2 : 3N/4]` | TP=8, rank=5, axis=0, requests `[5N/8 : 6N/8]` | Receiver pulls from same trainer rank (R2), takes upper half of R2's shard |
| `SHARD(0)` trainer TP=4 rank=1, range `[N/4 : N/2]` | TP=2, rank=0, axis=0, requests `[0 : N/2]` | Receiver pulls from **both** trainer R0 + R1, concatenates |

For cases where one inference rank needs slices from multiple trainer ranks (last row), the receiver picks **N candidates** instead of one:

```python
multi_source = receiver.discover_v2_sources_for_slice(
    model_name=...,
    target_slice=(start, end),
    shard_axis=0,
)
# Returns one SourceRef per trainer rank whose shard overlaps target_slice
# Receiver does N parallel RDMA pulls, concatenates in scratch
```

### Mixed-EP for MoE

Same machinery, on the expert axis. NemoRL v2 already does this for `owned_expert_ids`:

```python
candidates = receiver.discover_v2_sources(
    model_name=...,
    target_expert_ids_per_layer={
        5: {0, 1, 2, 3},    # this inference rank's owned experts in layer 5
        6: {0, 1, 2, 3},
    },
)
```

The picker matches candidates whose `expert_owner_per_rank` covers the needed experts (existing logic in `MxV2RefitReceiver.pick_best_source` — see `nemo_rl_v2.py`). For mixed-EP (trainer EP=4, inference EP=8), receivers may pull from multiple trainer ranks via the same `discover_v2_sources_for_slice` pattern.

### Why this matters for PrimeRL specifically

PI's `ExpertSlot` (in `slots.py`) and `build_topology()` (in `nixl_checkpoint_engine`-style code) already implement TP-matched and EP-matched pairing on the trainer side. They're computing `peer_chunk_descs` based on the publisher's known topology. **What's missing is the inverse**: when the inference layout differs from the trainer's, the receiver needs to express that. The compile-target registry + the slice-discovery API give it that vocabulary.

---

## 6. MX client adoption — layered with this plan

The two-phase migration from our earlier review of #2389 (Phase 1 surgical / Phase 2 client adoption) is the foundation this plan builds on:

### Phase 1 — surgical fixes against the `nixl_mx` in-tree code (drop-in patches)

The 6 inline-comment fixes (line numbers verified against `nixl_mx` HEAD `79ea824d8`):

1. Same-rank `add_remote_agent` filter in `transport_plan.py`
2. Freshest-per-rank dedup in `mx_rendezvous.py::wait_for_peers`
3. `HeartbeatThread` after `set_status(READY)` in `inference/.../nixl_mx.py`
4. Read timeout from config (not hardcoded 1200s)
5. MLA-guard for non-MLA models (`update_mla_absorbed_weights`)
6. HSDP barrier ordering in `trainer/.../nixl_mx.py`

These land **before** this plan starts — closes the bug classes without architectural change.

### Phase 2 — graduate the rendezvous half onto ModelExpress clients

Delete `src/prime_rl/transport/mx_rendezvous.py` (~185 LOC, replicates functionality already in `modelexpress`). Replace with imports of `MxV2TrainingPublisher` and `MxV2RefitReceiver`. The in-tree `NixlAgentWrapper` + `Slot` + `TransportPlan` + `classic_cuda_pool` stay — that's prime-rl-specific data-plane specialization and shouldn't move.

```diff
-from prime_rl.transport.mx_rendezvous import MxRendezvous
+from modelexpress import MxV2TrainingPublisher, MxV2RefitReceiver
```

This is what unblocks everything in §3-§5:

- `MxV2TrainingPublisher` exposes the v2 sidecar registry that §4's `compile_target` extends.
- `MxV2RefitReceiver.receive_weights_scratch` is the proven path from John's Dynamo work (380 Gbps GB300).
- `discover_v2_sources(compile_target_filter=...)` is a small extension to the existing picker.
- Heartbeat / freshest-dedup / retention all come along for free — no separate Phase 1 work needed once Phase 2 lands.

### Phase 3 (this plan) — compile-target registry + mixed-TP

- Add `compile_target` + `compile_metadata` to v2 shape registry (~30 LOC in `shape_descriptors.py` + `nemo_rl_v2.py`).
- Add `compile_target_filter` to `discover_v2_sources` (~15 LOC).
- Add `target_tp_layout` + `discover_v2_sources_for_slice` to `MxV2RefitReceiver` (~120 LOC).
- Add `compile_passes/` module in `src/prime_rl/inference/vllm/worker/` with `HFRaw`, `DeepGemmFP8`, `CutlassFP8` passes (~300 LOC). Or in NemoRL `nemo_rl/models/generation/vllm/compile_passes/` — see §7.
- PI's `nixl_mx.py` inference worker calls into the right `CompilePass` based on `engine.kernel_target`.

Total: ~450 LOC across MX + PrimeRL, all additive.

---

## 7. What we borrow from John Thompson's NemoRL+Dynamo work

Five specific pieces, all already proven on GB300 RoCE:

### 7.1 `receive_weights_scratch` is the foundation

John's path uses `MxRefitReceiver.receive_weights_scratch` because vLLM's `stacked_params_mapping` requires HF-named tensors that the receiver later passes to `model.load_weights()`. That's structurally identical to "trainer ships HF-raw, receiver compiles into kernel layout":

```python
# John's existing flow (NemoRL + Dynamo + vLLM v1)
weights = list(receiver._receiver.receive_weights_scratch(
    chosen.ref,
    timeout_seconds=mx_config.timeout_seconds,
    tensor_shapes=tensor_shapes,
))
self.model_runner.model.load_weights(weights=weights)   # vLLM does the HF→fused remap

# Our extension (PrimeRL post-#2389 + kernel compile)
weights = list(receiver._receiver.receive_weights_scratch(
    chosen.ref,
    timeout_seconds=mx_config.timeout_seconds,
    tensor_shapes=tensor_shapes,
    target_tp_layout=self._target_tp_layout,   # NEW
))
self._compile_pass.apply(weights, live_params)   # OUR compile dispatch
```

The mechanism is identical. Only the post-RDMA stage differs.

### 7.2 The `worker_extension_cls` injection pattern is cleaner than subclassing

John's `MxRefitWorkerExtension` (in `dynamo/vllm/mx_refit/extension.py`) is injected via vLLM v1's `parallel_config.worker_extension_cls`. The class has no `__init__`; vLLM merges its methods into the existing `Worker` via `__bases__`. State is stashed lazily on `self` with `_mx_` prefixed attribute names.

PI's `NIXLMxWeightUpdateWorker` today **subclasses** `Worker` directly. The extension-class pattern would let the refit logic live in a sibling module without touching the inheritance chain — useful when we add the compile passes (§3) because those want to live in their own package.

**Recommend**: when graduating PrimeRL to MX clients in Phase 2, also adopt the `worker_extension_cls` pattern for the inference worker. The two changes naturally compose.

### 7.3 PR #295's sidecar filter is required for any new v2 metadata

If we extend the v2 sidecar (§4) without keeping PR #295's filter in `MxRefitReceiver.receive_weights{,_scratch}`, the synthetic `__mx_v2_meta__` `TensorDescriptor` poisons `prep_xfer_dlist` again — same `NIXL_ERR_NOT_FOUND` John hit before May 22. **No new code needed; the filter is already in `kavink/nemo_rl_moe` HEAD `8594fd6`.** Just don't accidentally back it out.

### 7.4 The `FORCE_RDMA=1` test mode catches this class of bug in loopback

The v2 demo scripts (`scripts/v2_*_e2e_demo.py`) have a `FORCE_RDMA=1` env var (commit `e8e063b`) that pins `UCX_TLS` off `cuda_ipc` so intra-node loopback exercises the strict `rc_mlx5` descriptor-list validator. **Run every new compile-pass test under `FORCE_RDMA=1`** — otherwise we'll merge a sidecar / descriptor-list bug that doesn't show up until cross-node deploy.

### 7.5 The compile-pass module probably belongs in NemoRL first, mirrored into prime-rl

John's Dynamo path is the most mature target for testing new kernels (Qwen3-4B-Thinking GRPO smoke is already running cross-node at 380 Gbps). The compile passes themselves are framework-agnostic — they just take `(scratch_dict, live_params_dict)` and run torch ops. **Recommend**:

1. Implement `compile_passes/` first in `modelexpress_client/python/modelexpress/compile_passes/` — framework-neutral, reusable.
2. NemoRL + Dynamo path adopts it via `MxRefitWorkerExtension._compile_pass = HFRaw()` (or `DeepGemmFP8()`).
3. Validate end-to-end on GB300 with cutlass + DeepGemm kernels.
4. Mirror into PrimeRL's inference worker after Phase 2 graduates them to MX clients.

This sequence means we de-risk the compile-pass design on the path that's already shown working before we touch PrimeRL. Same play we ran for the v2 sidecar — designed in NemoRL, validated in NemoRL+Dynamo (John), then graduated to prime-rl.

---

## 8. Implementation phases

| Phase | Scope | Estimated LOC | Owner |
|---|---|---|---|
| **0** | Wait for #2389 to merge upstream | — | Matej |
| **1** | 6 surgical fixes against PI's `transport/*.py` + `inference/*.py` (closes bug classes) | ~100 LOC | Us — fast follow on Matej |
| **2** | Graduate `MxRendezvous` → `MxV2TrainingPublisher` / `MxV2RefitReceiver`; adopt `worker_extension_cls` pattern in inference worker | ~−400 LOC (PI's reimpl removed) + ~150 LOC import-and-call | Us |
| **3a** | Add `compile_target` + `compile_metadata` to v2 shape registry | ~30 LOC | Us |
| **3b** | Add `compile_target_filter` to `discover_v2_sources` | ~15 LOC | Us |
| **3c** | Add `target_tp_layout` + `discover_v2_sources_for_slice` to `MxV2RefitReceiver` | ~120 LOC | Us |
| **3d** | Implement `compile_passes/` (HFRaw, DeepGemmFP8, CutlassFP8) — in MX repo for reuse | ~300 LOC | Us |
| **3e** | Validate on NemoRL+Dynamo path (John's GB300 cluster) — Qwen3-4B-Thinking with DeepGemm and cutlass kernels both running on the same MX server | E2E | Us + John |
| **3f** | Mirror compile-pass dispatch into PI's `inference/vllm/worker/nixl_mx.py` | ~50 LOC | Us — PR back to PI |
| **4** | Mixed-TP / mixed-EP slice discovery wired end-to-end (multi-source RDMA pulls) | ~200 LOC | Us — separable from Phase 3 |

**Phases 0-1** are fully sequenced (must wait for upstream + apply surgical fixes). **Phases 2 onward** can run in parallel if we're willing to maintain a `kavink/post-2389-*` branch off PI's main + a follow-on PR per phase.

---

## 9. Open questions

1. **Does PI want the compile passes in their tree, or in MX?** If MX, they import a pluggable `CompilePassRegistry`. If their tree, the kernel ecosystem stays close to their Slot system (which already does fp8_blockwise). My lean: **MX**, because Dynamo + NemoRL also want them, and PI's per-Slot conversion stays for the trainer-side path when teams opt into "publish post-compile".

2. **Does cutlass-FP8 work on inference-side compute?** Compile pass needs ~200ms of CUDA time. If the inference engine is mid-rollout when the refit arrives, we either pause and run the compile or queue it for the next "between rollouts" window. PrimeRL's current orchestrator does the latter; this plan inherits.

3. **How do we handle trainers that publish post-compile (Matej's current path)?** Their `compile_target = "deep_gemm_fp8"`; receivers either accept it directly (fast path) or reject and look for `hf_raw`. Mixed fleets get clean error messages, not corrupt weights.

4. **Mixed-TP across nodes — what's the bandwidth math?** Trainer TP=4 ↔ inference TP=8 means each inference rank pulls from 1-2 trainer ranks. For Qwen3-30B-A3B on GB200 (~30 GB / 4 trainer ranks = 7.5 GB/rank), an inference rank pulling 2× 4 GB slices is well within NIC budget. For larger EP layouts where one inference rank needs experts from N>2 trainer ranks, fan-in becomes interesting — that's where pipeline replication (TensorHub) and rollouts-as-replicas pay off.

5. **What's the deprecation story for trainer-side compile?** We don't deprecate it — Matej's path stays valid for teams that want zero inference-side latency. The `compile_target` field is just informational; receivers filter on it.

6. **Should the compile pass run before or after `update_mla_absorbed_weights`?** After. MLA absorption operates on live params; compile runs first so live params are in the right layout when MLA absorption runs.

---

## 10. Component view + sequence diagram

```mermaid
flowchart TB
    subgraph trainer["Trainer side (after Phase 2)"]
        TBcast["NIXLMxWeightBroadcast<br/>(PI's lifecycle wrapper)"]
        TSlots["Slots: Sharded · Gathered · Expert<br/>(PI's data plane, kept)"]
        TPlan["TransportPlan<br/>(PI's, kept)"]
        TPub["<b>MxV2TrainingPublisher</b><br/>(NEW — replaces MxRendezvous)"]
        TAgent["NixlAgentWrapper<br/>(PI's, kept)"]
        TBcast --> TSlots
        TBcast --> TPlan
        TBcast --> TPub
        TSlots --> TAgent
        TPlan --> TAgent
        TPub -. publishes registry incl.<br/>compile_target=hf_raw .-> TPlan
    end

    subgraph mx["MX control plane (unchanged)"]
        MXSVR[("MX Server · gRPC + Redis<br/>shape registry, compile-target<br/>filter, tree fan-out catalog")]
    end

    subgraph inf["Inference side (after Phase 2+3)"]
        IRec["<b>MxV2RefitReceiver</b><br/>(NEW — replaces ad-hoc rendezvous)"]
        IScratch["receive_weights_scratch<br/>+ target_tp_layout<br/>(extended John's path)"]
        ICompile["<b>CompilePass dispatch</b> (NEW)<br/>HFRaw · DeepGemmFP8 · CutlassFP8"]
        ILive["vLLM model.named_parameters()<br/>(live params — kernel-specific layout)"]
        IRec --> IScratch
        IScratch --> ICompile
        ICompile --> ILive
    end

    TPub <-.->|"publish_metadata (incl. compile_target)<br/>set_status · update_status"| MXSVR
    IRec <-.->|"discover_v2_sources(compile_target_filter=...)<br/>list_sources · get_metadata"| MXSVR
    TAgent <==>|"one-sided RDMA WRITE<br/>UCX rc_mlx5 / RoCE<br/>(HF-raw bytes, post-cast pre-compile)"| IScratch

    style mx fill:#fec,stroke:#963
    style MXSVR fill:#fec,stroke:#963,stroke-width:2px
    style TPub fill:#cce,stroke:#33c,stroke-width:2px
    style IRec fill:#cce,stroke:#33c,stroke-width:2px
    style ICompile fill:#cfc,stroke:#363,stroke-width:2px
    style ILive fill:#fcc,stroke:#c33
```

### One refit cycle, after this plan lands

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator
    participant T as Trainer (NIXLMxWeightBroadcast)
    participant MX as MX Server
    participant I as Inference worker
    participant C as CompilePass

    Note over T,I: BOOT (once per refit run)
    O->>I: POST /init_nixl_mx (host, port, rank, kernel_target=deep_gemm_fp8)
    I->>I: register live params with NIXL (PI's data plane)
    I->>MX: publish_metadata(role=inference, kernel_target=deep_gemm_fp8)
    I->>MX: update_status(READY)

    Note over T,I: PER REFIT STEP
    O->>I: POST /update_weights
    I->>MX: wait_for_all_peers_ready(role=trainer, READY)

    T->>T: lazy_init slots; per-rank scratch fill from state_dict<br/>(NO trainer-side cutlass — bytes are HF-raw)
    T->>MX: publish_metadata(role=trainer, compile_target="hf_raw",<br/>compile_metadata={...}, shape registry per tensor)
    T->>MX: update_status(INITIALIZING → READY)

    I->>MX: discover_v2_sources(model, min_version=N,<br/>compile_target_filter={"hf_raw"},<br/>target_tp_layout=TP=8 rank=3)
    MX-->>I: candidates = [trainer R0 (covers requested slice)]
    I->>I: pick_best_source

    Note right of I: SCRATCH PATH (from John's NemoRL+Dynamo work)
    I->>T: NIXL one-sided RDMA WRITE → scratch buffers<br/>(HF-raw layout, ~380 Gbps on GB300 RoCE)
    I->>I: torch.cuda.synchronize

    Note right of I: COMPILE PASS (new — runs inference-side, ~50-200ms)
    I->>C: apply(scratch_buffers, live_params)<br/>e.g. DeepGemm scale interleave, fused gate_up_proj pack
    C-->>I: live params updated in DeepGemm-friendly layout

    I->>I: update_mla_absorbed_weights (if MLA model)
    I->>MX: publish_self_as_source(role=inference_replica, version=N)<br/>(tree fan-out for next refit)

    I-->>O: 200 OK
    O->>O: scheduler advances · next rollout uses new weights
```

---

## 11. Cross-references

ModelExpress design docs (NVIDIA-authored, for context on the client surface this plan adopts):

- [`docs/RL/PRIMERL_MX_OVERVIEW.md`](https://github.com/ai-dynamo/modelexpress/blob/main/docs/RL/PRIMERL_MX_OVERVIEW.md) — the foundational prime-rl × MX integration design (catalog + star wiring story).
- [`docs/RL/NEMORL_MX_OVERVIEW.md`](https://github.com/ai-dynamo/modelexpress/blob/kavink/nemo_rl_moe/docs/RL/NEMORL_MX_OVERVIEW.md) — the v2 design (rank-to-rank, tree fan-out, expert filter, shape registry) that this plan extends with the compile-target axis.
- [`docs/RL/VERL_MX_OVERVIEW.md`](https://github.com/ai-dynamo/modelexpress/blob/main/docs/RL/VERL_MX_OVERVIEW.md) — the verl `MxCheckpointEngine` integration; sibling adopter of the same MX clients.

Upstream branches this plan refers to:

- ModelExpress branch [`kavink/nemo_rl_moe`](https://github.com/ai-dynamo/modelexpress/tree/kavink/nemo_rl_moe) — the v2 client surface (`MxV2TrainingPublisher`, `MxV2RefitReceiver`), `shape_descriptors`, sidecar transport, and PR #295's sidecar filter. The MX-side dependency that Phase 2 imports.
- ModelExpress PR [#295](https://github.com/ai-dynamo/modelexpress/pull/295) — synthetic-sidecar `TensorDescriptor` filter in `MxRefitReceiver`. Required for any v2 metadata extension (including this plan's `compile_target` field) to survive `prep_xfer_dlist` validation on cross-node RoCE. Already merged into `kavink/nemo_rl_moe`; just don't back it out.
- NemoRL × Dynamo branches (John Thompson, NVIDIA): [`KavinKrishnan/RL:kavink/mx_integration`](https://github.com/KavinKrishnan/RL/tree/kavink/mx_integration) (NeMo-RL side) + the Dynamo-side companion. Validated at 380 Gbps on GB300 RoCE for an 8.82 GB / 399-tensor refit (Qwen3-4B-Thinking GRPO smoke).

NVIDIA-internal context (not necessary for upstream review, listed for our own bookkeeping):

- The 6 inline review comments + summary message we have queued for #2389 — verified line numbers against HEAD `dabaa19f5` (still applicable on `79ea824d8` after I re-checked May 27). Phase 1 of this rollout.
- Current state of #2389: +10 commits since `dabaa19f5` (4× conversion-cast polish, 4× DeepGemm env-var hygiene, 2× config/import fixes). None touched the 6 flagged lines.
