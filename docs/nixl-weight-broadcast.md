# NIXL weight broadcast

End-to-end, step-by-step walkthrough of how the trainer sends updated
weights to the inference engines over NVIDIA NIXL (UCX + IB RDMA). Target
audience: anyone modifying the weight-transfer path or debugging a run.

## Actors and topology

Prod layout (12-node disaggregated, what we have been iterating on):

| Role | Nodes | GPUs | Parallelism |
|---|---|---|---|
| Trainer | 8 | 64 | `ep=8`, `dp_shard_mod_ep=8` (FSDP inside each EP group). `trainer_ws = 64` |
| Inference — prefill | 2 | 16 | `dp=16`, EP replicated |
| Inference — decode | 2 | 16 | `dp=16` with expert sharding via vLLM's `expert_map` |

`inference_ws = 32` (prefill + decode); `total_ws = trainer_ws + inference_ws = 96`.

Everything below flows through `NIXLWeightBroadcast` on the trainer side and
`NIXLWeightUpdateWorker` on the inference side, with a
`vllm.distributed.utils.StatelessProcessGroup` (SPG) serving as the 96-rank
rendezvous channel.

## Slot abstraction

The trainer's FSDP parameter buffers and vLLM's packed kernel tensors do
not line up byte-for-byte: FSDP shards along dim 0, keeps experts per-rank,
and stores weights as bf16; vLLM fuses multiple source tensors into one
destination (e.g. `q_a_proj + kv_a_proj_with_mqa → fused_qkv_a_proj`),
FP8-quantizes most projections, and keeps some layernorm affine params in
fp32.

We bridge this with **slots**: per-layer, contiguous destination buffers
on the trainer that (a) match the vLLM-side layout the NIC will write into,
and (b) are individually registrable with NIXL. One `allocate_slots` call
produces all slots; they are reused across every broadcast.

### ConversionSpec + QuantizationSpec

The shape, dtype, and transformation for each slot are described by two
model-agnostic dataclasses in
`src/prime_rl/trainer/models/conversion_spec.py`:

- **`QuantizationSpec(destination_dtype, scale_suffix="")`** — the
  per-slot *transformation*. `destination_dtype` is what the slot is
  allocated in. `scale_suffix` drives two things: whether a paired FP8
  scale buffer is allocated, and what suffix replaces `.weight` /
  `_weight` in the destination name to form the scale buffer's name.
  Non-empty `scale_suffix` → FP8-quantize path; empty → plain
  `out.copy_(src)` that auto-casts to `destination_dtype`.
  `QuantizationSpec.apply(src, out, scale_out)` is the one entry point;
  it dispatches to `fp8_block_quantize` (2D) or
  `grouped_fp8_block_quantize` (3D stacked-expert) based on `src.ndim`
  when a scale is required, and to `copy_cast` otherwise.

- **`ConversionSpec(dst, sources, cat_dim, quantization)`** — the
  *routing*. `dst` is the vLLM destination suffix (after
  `model.layers.{i}.`), `sources` are one or more trainer-side source
  suffixes fused along `cat_dim`, and `quantization` is a
  `QuantizationSpec`. Default `quantization` is bf16 copy. `scale_name`
  and `per_source_scale_key` derive the scale buffer's full name from
  the destination or per-source slot key respectively.

Per-model spec tables (e.g.
`src/prime_rl/trainer/models/glm_moe_dsa/converting_glm_moe_dsa.py`)
list one `ConversionSpec` per vLLM destination tensor. Overrides for
layernorm fp32 dtype or FP8 projections are explicit:

```python
ConversionSpec("self_attn.indexer.k_norm.weight", (...), quantization=QuantizationSpec(torch.float32))
ConversionSpec("self_attn.o_proj.weight", (...), quantization=QuantizationSpec(torch.float8_e4m3fn, ".weight_scale_inv"))
```

Two kinds of slot on the trainer side, both driven by the same specs:

- **Expert slot** — one fused buffer per layer per expert spec. Shape
  `(num_local_experts, cat_dim_size, hidden)`. Built by concatenating
  each local expert's `w1+w3` or `w2` along `cat_dim=1`, then calling
  `spec.quantization.apply` which picks the 3D grouped quantize kernel.
  Scale buffer shape is
  `(num_local_experts, ceil(rows/128), ceil(cols/128))` fp32.

- **Non-expert slot** — one buffer per source tensor. Shape driven by
  `per_shard` vs `gather` handling (below); dtype is `spec.slot_dtype`
  (= `spec.quantization.destination_dtype`). Scale buffer name is
  `spec.per_source_scale_key(slot_key)` when the spec is quantized.

Non-expert slots use one of two **handling modes**:

- **per_shard** — slot holds this rank's FSDP shard (`rows = src_rows /
  fsdp_total`). Every trainer rank writes its shard into `chunk[fsdp_rank]`
  of the inference-side full tensor, on every inference peer. Used for
  tensors where
  1. `src_rows % fsdp_total == 0`, and
  2. for quantized specs, `(src_rows / fsdp_total) % 128 == 0` (FP8 block
     alignment), and
  3. the source tensor is at least `_SMALL_NON_EXPERT_BYTES` (2 MiB
     currently).
- **gather** — slot holds the full tensor. One trainer rank writes the
  whole slot once per inference peer, round-robin across trainer ranks by
  `i % trainer_ws == my_rank`. Everything that fails the per_shard test
  (awkward shape, too small to bother sharding) ends up here.

The threshold matters: for a 128-element layernorm, per-shard fans out to
`fsdp_total × inference_ws = 2048` writes; gather drops it to `~32`. On
GLM-5 we cut total handle count by ~60% just by gathering anything below
2 MiB.

## Phase 1 — setup

Triggered by `setup_weight_broadcast(config.type == "nixl", …)` at
trainer init. Performed once per process.

1. **Allocate slots** (`model.allocate_slots(parallel_dims)`). Runs inside
   `with classic_cuda_alloc():` so every slot is a contiguous
   `cudaMalloc` block — see [Platform gotchas](#platform-gotchas) for why.

2. **Compute layouts** (`model.non_expert_slot_layout(parallel_dims)`).
   For every non-expert slot key returns
   `{inference_name, offset_rows, rows, handling}`. This is what the
   inference side uses in round 1 to know where each slot lands.

3. **Build NIXL agent** (`NixlAgentWrapper`). Wraps a UCX-backed
   `nixl_agent`. `pin_ucx_rail(local_rank)` hard-sets
   `UCX_NET_DEVICES` to the GPU's PIX-attached NIC so inference decode's
   pre-set `UCX_NET_DEVICES=mlx5_0:1` (used by vLLM's KV `NixlConnector`)
   does not bottleneck every WRITE through a single NIC on each decode
   node.

4. **Register slots**. For each slot: `register_tensor` pins its memory
   for RDMA and produces an xfer descriptor list. `chunked_descs(slot,
   num_local_experts)` splits expert slots along the leading dim;
   non-expert slots use `chunked_descs(slot, 1)`. Prepping these via
   `prep_local` yields the handles we pass to `make_prepped_xfer` later.

5. **SPG rendezvous — round 1** (all 96 ranks).
   - Trainer publishes `{role: "trainer", global_rank, non_expert_layout}`.
   - Inference publishes `{role: "inference", global_rank, expert_map}`
     where `expert_map[prefix]` is the list of global expert IDs this
     inference rank holds for that MoE prefix.
   - Agent metadata is NOT yet exchanged. The inference side has to
     register its chunks first so the metadata it ships in round 2
     carries every MR the trainer is going to write into.

6. **Inference builds chunked descriptors**. Using the trainer's
   `non_expert_layout`, each inference rank takes its single already-
   registered full tensor (e.g. `model.layers.12.self_attn.q_a_proj.weight`),
   narrows it to `(offset_rows, rows)`, and further narrows into
   `trainer_ws` equal chunks for `per_shard` or one chunk for `gather`.
   For each sub-range it calls `get_xfer_descs(...)` and serializes the
   resulting 1-entry xfer dlist. Expert tensors are chunked into
   `num_local` experts.

   Why this matters: the serialized descs carry the rkey bound to
   inference's UCX backend. Building descriptors on the trainer from raw
   `(ptr, size, device)` tuples loses the rkey → the NIC rejects the
   WRITE at landing time.

7. **SPG rendezvous — round 2**.
   - Trainer publishes fresh `agent_metadata` (now includes every chunk
     registration done in step 4).
   - Inference publishes fresh `agent_metadata` plus
     `descriptors: {slot_name → [serialized chunk dlist, …]}` plus its
     `expert_map`.

8. **Trainer assembles the write table**. For every one of its slots,
   for every inference peer:
   - **Expert slot**: look up the peer's `expert_map[moe_prefix]`, and
     only write a local expert if the peer owns it. Both prefill and
     decode are EP-sharded (`enable_expert_parallel = true`), so each
     global expert lives on exactly one prefill peer and one decode
     peer — not on every peer. Record
     `(local_prep, local_idx, remote_prep, remote_idx, peer, tag)`.
   - **Non-expert `per_shard`**: target `peer_descriptors[slot][my_rank]`
     on every inference peer. Record the 5-tuple.
   - **Non-expert `gather`**: target `peer_descriptors[slot][0]`, but
     only on peers where `peer_index % trainer_ws == my_rank`
     (round-robin).

   Size-mismatch sanity check runs here (`_check`) — we compare local
   slot bytes vs the remote dlist's reported size and log `SIZE
   MISMATCH` before the first post. This is what caught the bf16↔fp32
   dtype bugs on layernorms and expert_bias.

9. **Eager connect**. For each inference peer
   `self._agent.make_connection(peer_name)` so UCX opens its RC-mlx5
   endpoint up front rather than during the first WRITE.

At this point the trainer has a static `self._writes` list of roughly
9k–12k entries per rank, plus 64 eager UCX endpoints to each inference
rank. Bytes per push per rank is ~25.9 GB for GLM-5.

## Phase 2 — push_once (per training step)

Happens inside `broadcast_weights()` whenever the trainer decides to
send an update.

1. **Orchestrator protocol (rank 0 only)**. Rank 0 touches a `STABLE`
   marker under `broadcasts/step_N/`. The orchestrator loop sees it,
   calls `/pause?mode=keep&clear_cache=false` on every inference
   admin endpoint, waits for all to ack, then drops `NIXL_READY` in the
   same directory. Pause is what makes it safe to RDMA-WRITE into a
   live vLLM engine's parameter buffers — no kernel is using them while
   pause is held.

2. **Trainer-wide `dist.barrier()`**. Without this, non-master trainer
   ranks would start posting WRITEs before `NIXL_READY` is dropped, i.e.
   before inference has actually paused. The barrier ensures all 64
   trainer ranks enter `push_once` together.

3. **Convert every layer into its slots** (`convert_layer_to_vllm_kernel`
   for all 78 layers). This is where:
   - Experts: concatenate `w1+w3` for `w13_weight`, concatenate `w2` to
     itself (1 source) for `w2_weight`, then FP8-block-quantize directly
     into the pre-allocated slot (3D `grouped_fp8_block_quantize`).
   - Non-expert `per_shard`: source DTensor's `to_local()` slab is
     copied or FP8-quantized into the slot. `copy_` auto-casts bf16→fp32
     for the layernorm/bias slots.
   - Non-expert `gather`: source's `full_tensor()` (or plain tensor) is
     copied/quantized into the full-shape slot. Every trainer rank
     materializes the full value even though only one will send it —
     cost is negligible for sub-2 MiB tensors.
   - `torch.cuda.synchronize` at the end so the slots are populated
     before any WRITE starts.

4. **Chunked drain**. Rather than post all ~10k writes and then wait,
   we interleave:
   ```python
   for i, (lp, li, rp, ri, peer, tag) in enumerate(self._writes):
       handles.append(self._agent.post_write(lp, li, rp, ri))
       if (i + 1) % flush_every == 0:
           drain(handles)        # busy-wait each to DONE, then clear
   drain(handles)                # tail
   ```
   `flush_every=100`. Draining keeps UCX's RC send queue bounded —
   posting 10k at once previously wedged UCX without any bytes leaving
   the NIC. Each drain logs `drained through N/total in Xms` so we can
   see progress.

5. **SPG `barrier()` at end of push_once**. All 64 trainer ranks plus
   all 32 inference ranks (inference enters the barrier as the first
   line of `update_weights_from_path`) hit this. When it releases, the
   trainer knows every WRITE has been acknowledged by the peer and
   inference knows all weights are now in place. Orchestrator then
   calls `/resume`.

6. **`update_mla_absorbed_weights`** fires on inference after the
   barrier — it re-runs the MLA absorption trick for the freshly
   written attention weights. Nothing NIXL-specific about it.

Steady-state on prod: ~4.3 s per broadcast (2.2 s convert, 1.2 s
post+drain, 0.8 s barrier), ~7.3 GB/s wire / ~20 GB/s fan-out.

## Non-expert flow — why the handle count matters

`bytes_per_push = 25.9 GB` fans out to **every** inference rank for
non-expert tensors (since those are replicated), so total traffic per
broadcast is `25.9 × 32 ≈ 830 GB` on the receive side. What we actually
pay in latency is dominated by **handle count**, not bytes, because
each WRITE is ~10s of µs of per-handle overhead for small tensors.

The 2 MiB gather threshold exists for exactly this reason: a 128-element
layernorm is 256 B, sharded across 64 FSDP ranks it becomes 4 B per
shard, then WRITE'd to 32 inference peers by each of 64 trainer ranks.
Without the threshold that single tensor costs 2048 handles for 8 KiB
of actual data. With the threshold it costs ~32 handles for the same
8 KiB. For the ~30 small non-expert tensors per layer × 78 layers
that's the difference between 50k and 5k handles total.

## Expert flow — why the fan-out is already small

Experts are sharded on both sides: the trainer owns
`num_local_experts = num_experts / (ep_size × fsdp_inner)` local
experts, and each inference rank owns some subset of the 256 global
experts per MoE layer (vLLM's expert-parallel `expert_map`).

Per expert slot per trainer rank, we write once per **inference rank
that owns that global expert**:

```
for local_idx, global_expert in enumerate(owned_global_experts):
    for peer in inference_peers:
        if global_expert in peer.expert_map[moe_prefix]:
            write(local_idx → peer[remote_idx_for_that_expert])
```

With `enable_expert_parallel = true` on both prefill and decode, every
global expert lives on exactly one prefill peer and one decode peer.
Net handle count: each trainer-local expert generates two writes per
expert slot per layer, so total expert writes scale with
`local_experts × 2 × slot_keys × layers` — linear in EP degree on the
inference side, not DP. That's why the non-expert per_shard fan-out
dominates the total handle count.

## Platform gotchas (the non-obvious bits)

All of these cost a full debug session to find and are now permanent
in the code:

1. **Classic `cudaMalloc` pool for slots**. With
   `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` torch hands out
   `cuMemCreate`+`cuMemMap` VMM-backed VA. `ibv_reg_mr` accepts the VA
   but the mlx5 HCA's MMU walk at WRITE-landing time fails with
   `syndrome 0x4` (Local Protection), because `nvidia_peermem` can't
   pin VA that spans multiple physical handles. UCX then tears the
   endpoint down and the error surfaces as `NIXL_ERR_REMOTE_DISCONNECT`.
   Fix: `classic_cuda_alloc()` scopes slot allocation into a
   `CUDAPluggableAllocator` + `MemPool` that calls `cudaMalloc` /
   `cudaFree`. Everything else in the process keeps expandable
   segments.

2. **`libcudart.so` `RTLD_GLOBAL` preload**. TileLang ships a stub
   libcudart that proxies via `dlsym(RTLD_DEFAULT, …)`. If the stub
   wins the dlsym race (no real libcudart loaded globally yet), its
   self-check fails and it `abort()`s the process the moment we enter
   the MemPool context. The preload has to run *before* `import
   tilelang`, so it lives at the top of both
   `src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py` and
   `.../sparse_mla_bwd.py` (the only two files that import tilelang),
   wrapped in `try/except` for CI hosts without a CUDA runtime.

3. **Per-GPU PIX NIC pinning**. GPUs 4–7 have three PIX-attached NICs
   (mlx5_6, mlx5_7, mlx5_8), and `mlx5_8` is DOWN on every trainer node
   in this cluster. UCX with `UCX_NET_DEVICES=all` picks `mlx5_8` for
   some rank-4 ↔ peer endpoints and hits REMOTE_DISCONNECT. Fix:
   `pin_ucx_rail(local_rank)` hard-sets `UCX_NET_DEVICES` to the GPU's
   first PIX-attached NIC (from `nvidia-smi topo -m`).

4. **Expandable segments on the trainer is safe**. Because of fix #1.
   Prior to the MemPool carve-out we had to export
   `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"` on the trainer
   under NIXL, which lost the fragmentation-mitigation benefit for the
   full ~200 GB of FSDP params + activations.

5. **Inference decode `UCX_NET_DEVICES=mlx5_0:1`**. vLLM's PD KV
   `NixlConnector` sets this so its UCP worker pins to `mlx5_0`. We
   cannot unset it process-wide, but our weight-transfer
   `NixlAgentWrapper` is created *after* the PD connector, and
   `pin_ucx_rail` now does a hard override of `UCX_NET_DEVICES` before
   creating the new UCX agent. The PD connector already has its own
   worker up with its own env snapshot; the new NIXL agent we build
   for weights sees the per-GPU PIX NIC. Result: ~7.5 GB/s wire /
   ~20 GB/s fan-out vs ~4.8 GB/s / ~10 GB/s when the receive side was
   bottlenecked on one NIC per decode node.

6. **fp32 slot dtype for three specific sources**. vLLM keeps
   `self_attn.indexer.k_norm.weight`, `self_attn.indexer.k_norm.bias`,
   and `mlp.expert_bias` in fp32. Trainer defaults to bf16 for
   non-quantized params → 2× byte-length mismatch → `makeXferReq:
   length mismatch at index pair 0` at post time. Expressed in the
   spec table as `quantization=QuantizationSpec(torch.float32)`;
   `QuantizationSpec.apply` does `out.copy_(src)` which auto-casts
   bf16→fp32.

7. **Dist barrier before push_once** (section 2 step 2). Not cosmetic:
   without it non-master trainer ranks start RDMA-writing into
   inference memory before inference has paused, which corrupts
   whatever forward pass is still running.

8. **mlx5_8 DOWN node exclusion**. Prod still lists
   `ltc-idc3-hgx8-h200-2` and `ltc-idc3-hgx8-h200-65` in the sbatch
   `exclude` list because those nodes have `EnableStreamMemOPs=0` and
   empty `RegistryDwords` (no IBGDA). UCX fails to start on them
   entirely. Unrelated to the other fixes but part of the recipe.

## What we still don't do

- No background weight-transfer. `push_once` is synchronous; rollouts
  pause while it runs. Orchestrator pauses inference anyway so this is
  not free bandwidth we are losing.
- No delta-only broadcast. We always ship the full 25.9 GB even if
  most experts didn't move. Worth considering for some RL setups.
- No per-rank stream concurrency in the drain loop. Each rank drains
  serially on its main CUDA stream. Could be faster but UCX already
  saturates the NIC at ~7.5 GB/s, so the headroom is unclear.
- Stragglers: local ranks 1–7 arrive at the SPG barrier consistently
  ~80–200 ms after local rank 0. Likely NUMA 1 / remote-socket NIC hop.
  Currently accepted.

## File map

| Path | Role |
|---|---|
| `src/prime_rl/trainer/rl/broadcast/nixl.py` | Trainer-side `NIXLWeightBroadcast`. Sets up the agent, does the 2-round rendezvous, builds the write table, runs `push_once`. |
| `src/prime_rl/inference/vllm/worker/nixl.py` | vLLM worker extension. Registers params+scale buffers once, publishes chunk descriptors, barriers + resumes on each `update_weights_from_path`. |
| `src/prime_rl/utils/nixl_transfer.py` | `NixlAgentWrapper`, `pin_ucx_rail`, `map_gpu_to_nic`. NIXL/UCX knobs live here. |
| `src/prime_rl/utils/classic_cuda_pool.py` | JIT-compiled `CUDAPluggableAllocator` + `MemPool` + `classic_cuda_alloc()` context manager. |
| `src/prime_rl/trainer/models/conversion_spec.py` | `ConversionSpec` + `QuantizationSpec`. Model-agnostic; per-model spec tables import these. |
| `src/prime_rl/trainer/models/kernels/sparse_mla_{fwd,bwd}.py` | TileLang kernels; libcudart RTLD_GLOBAL preload lives at the top of these (before `import tilelang`). |
| `src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py` | `allocate_slots`, `non_expert_slot_layout`, `convert_layer_to_vllm_kernel`. Model-specific: size threshold, fp32 source set, expert cat dims. |
| `src/prime_rl/trainer/models/glm_moe_dsa/converting_glm_moe_dsa.py` | `_Spec` table — the source of truth for which source tensors fuse into which vLLM destination, and which are quantized. |
| `src/prime_rl/templates/multi_node_rl.sbatch.j2` | Trainer env (`UCX_TLS`, `expandable_segments:True`, no `UCX_NET_DEVICES=all`). Inference decode env (`UCX_NET_DEVICES=mlx5_0:1` for vLLM's PD connector). |
