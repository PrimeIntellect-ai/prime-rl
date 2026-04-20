# NIXL weight broadcast — system contract

What each role promises, what flows between them, and what the system
guarantees after a push.

## Roles

| Role | What it runs | What it owns |
|---|---|---|
| **Orchestrator** | `prime_rl.orchestrator.orchestrator` | Pause/resume of the inference pool, `STABLE`/`NIXL_READY` markers, train-step pacing. |
| **Trainer** | `NIXLWeightBroadcast` → `TransportPlan` | Source of truth for model weights. Decides when a broadcast happens, drives the transfer. |
| **Inference** | `NIXLWeightUpdateWorker` per vLLM worker | Destination buffers. Pauses forward pass during a broadcast, resumes only when the orchestrator allows. |

The transfer happens end-to-end over:

* **SPG** (TCP) — rendezvous, barriers. `trainer_ws + inference_ws`
  ranks, established once at trainer init.
* **NIXL / UCX / IB RDMA** — the data path. Trainer posts WRITEs into
  pre-registered inference parameter buffers.
* **Filesystem markers** — one-way orchestrator ↔ trainer signaling
  (`STABLE`, `NIXL_READY`).

## Trainer ↔ inference contract

The trainer and inference agree on three things *before the first push*
and never renegotiate:

1. **The slot inventory.** Every trainer-side destination buffer has a
   unique `slot_key`. The inference side publishes a descriptor list
   under the same key. Expert slots use the destination param name;
   non-expert slots use the source-tensor name.
2. **The layout of every non-expert destination.** Trainer ships one
   `LayoutEntry(slot_key, inference_name, offset_rows, rows, num_chunks)`
   per slot-buffer in SPG round 1. Inference narrows its vLLM tensor
   per those coordinates and publishes one serialized xfer dlist per
   chunk.
3. **The expert map.** Inference publishes
   `{moe_prefix: [global_expert_id, …]}` so the trainer knows which
   peers own which global experts. Trainer only writes a local expert
   to peers that own it.

Once the write table is built, every broadcast reuses it.

### Per-push guarantees (what `push_once` provides)

After `push_once` returns **on every trainer rank** and
`update_weights_from_path` returns **on every inference rank**:

* Every inference-side parameter buffer that the trainer is responsible
  for has been overwritten with the current step's weights (after
  quantization + dtype cast as declared by the slot's `QuantizationSpec`).
* All RDMA WRITEs have been acknowledged at the remote NIC; no writes
  are in flight.
* MLA absorbed weights (`W_UV`, `W_UK_T`) on inference have been
  recomputed from the freshly written `kv_b_proj`.

### Per-push non-guarantees

* **No freshness beyond the current step.** If the trainer updates
  weights again before the next barrier lands, inference may observe
  a mixed snapshot. The orchestrator's pause/resume is what makes this
  safe in practice.
* **No delta.** Every push ships the entire registered surface,
  regardless of which params changed.
* **No ordering between slots.** Writes are posted in a fixed order but
  drained in batches; an inference observer that isn't paused would
  see torn writes.

## Orchestrator ↔ trainer ↔ inference contract

Per step, the orchestrator is the one authority that says "it's safe to
overwrite inference weights now" and "you can start serving again."

```
trainer rank 0           orchestrator              inference (all ranks)
     │                         │                         │
     ├── touch STABLE ─────────▶                         │
     │                         ├── /pause ───────────────▶
     │                         │◀── ack all ─────────────┤
     │                         ├── touch NIXL_READY      │
     │◀── see NIXL_READY ──────┤                         │
     │                                                   │
     ├───────── dist.barrier() across all trainer ranks ─│
     │                                                   │
     ├─────────── RDMA WRITEs (every rank) ─────────────▶│
     │                                                   │
     ├──────── spg.barrier() across trainer+inference ───│
     │                                                   │
     │                         │◀── /resume ─────────────┤
     │                         ├── resume ──────────────▶│
```

The contract:

* **Trainer promises:** no rank posts any RDMA WRITE before
  `NIXL_READY` is observed. The `dist.barrier()` across all trainer
  ranks enforces this — otherwise non-master ranks would race ahead.
* **Orchestrator promises:** once `NIXL_READY` is written, every
  inference worker has paused; no forward pass is reading params.
* **Inference promises:** once `update_weights_from_path` enters its
  SPG barrier, its params are quiescent and remain quiescent until
  both the barrier releases and `/resume` returns.
* **Shared ack:** the final `spg.barrier()` at the end of `push_once`
  is the single synchronization point that gates "weights are now in
  place" across the 96-rank cluster.

## Registration invariants (set once, forever)

These are properties of the pre-registered buffers. Breaking them
causes `NIXL_ERR_INVALID_PARAM` at post time or mlx5 Local Protection
at WRITE-landing time — both are debugging sinkholes.

* **One MR per logical buffer on inference.** The full vLLM tensor is
  registered once. Per-chunk xfer descriptors resolve to that MR's
  rkey at write time. Registering overlapping per-chunk MRs trips
  mlx5 rkey lookup.
* **Trainer slots live in the classic cudaMalloc pool.** Not in
  PyTorch's VMM `expandable_segments` pool — `nvidia_peermem` refuses
  cuMemMap-backed VA. Managed by `classic_cuda_alloc()` context.
* **NIC pinning is per-GPU.** Every trainer agent uses its GPU's
  PIX-attached NIC via `pin_ucx_rail(local_rank)`. Without this, inference
  decode's pre-set `UCX_NET_DEVICES=mlx5_0:1` (from vLLM's PD KV
  connector) would serialize every weight write through one NIC per
  decode node.
* **Chunk selection is prep-time, not post-time.** Each `remote_prep`
  is built from exactly one serialized dlist entry. `post_write` uses
  `remote_idx=0` always; the chunk is already encoded in the prep
  itself.

## What flows on the wire

### SPG control plane (rendezvous, once)

Round 1: layout only — trainer ships `list[LayoutEntry]`, inference
ships `expert_map`. Agent metadata is deferred so round-2 metadata
covers every chunk MR.

Round 2: agent metadata + inference's `descriptors` (per-slot_key
lists of serialized chunk dlists) + `expert_map` again.

### NIXL data plane (every push)

One RDMA WRITE per `(local_slot_chunk, inference_peer_chunk)` pair.
Write table size is fixed at rendezvous; per-push the only thing that
changes is the bytes.

### SPG control plane (every push)

Exactly one barrier at the end of `push_once`, joined by all trainer
and inference ranks.

### Filesystem (every push)

One `STABLE` touched by trainer rank 0, one `NIXL_READY` touched by
the orchestrator, under `broadcasts/step_N/` in the run's output dir.

## Who can break the contract

* **Changing a `ConversionSpec` between runs** (dtype, sources, cat_dim)
  without rebuilding the write table on both sides — the slot inventory
  and layout no longer match.
* **Allocating slots outside `classic_cuda_alloc()`.**
* **Creating the trainer's `NixlAgentWrapper` before `pin_ucx_rail`.**
* **Posting WRITEs before `NIXL_READY` / before the trainer-side
  `dist.barrier()`** — races against live forward passes.
* **Skipping the end-of-push `spg.barrier()`.** Orchestrator will
  `/resume` inference before some peers have acked their writes.
* **Registering the same inference tensor twice.** Overlapping MRs
  are what `makeXferReq` refuses with LOCAL_PROTECTION.

## Scope boundary

Not part of the contract:

* How `ConversionSpec` is constructed from model code — that's the
  model's business (`conversion_specs(layer_idx)` hook).
* Which UCX backends / transports are selected — `NixlAgentWrapper`
  picks them based on env vars set by `pin_ucx_rail`.
* How FSDP / EP / CP meshes are built — `ParallelDims` is handed to
  `TransportPlan`, the plan reads the mesh but does not shape it.
* How inference's `expert_map` is computed — `build_expert_map`
  reads it off the vLLM MoE modules.
* Orchestrator pause/resume internals — the trainer-side code only
  waits for the `NIXL_READY` marker.
