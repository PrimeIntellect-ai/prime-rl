# NIXL weight broadcast — architecture contract

Who owns what, when it's created, and what's on the wire. For the deeper
design narrative see `docs/nixl-weight-broadcast.md` (partially stale).

## Layering

```
  ┌─────────────────────────── trainer process ──────────────────────────────┐
  │                                                                          │
  │  NIXLWeightBroadcast                                                     │
  │  ├─ NixlAgentWrapper            (UCX / RDMA)                             │
  │  ├─ StatelessProcessGroup       (TCP rendezvous, 2 rounds)               │
  │  └─ TransportPlan                                                        │
  │     ├─ list[Slot]               (trainer-side buffers, built once)       │
  │     ├─ _local_preps             (per-slot-buffer NIXL prep dlist)        │
  │     └─ list[_ResolvedWrite]     (fully-resolved write table)             │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
                ▲                                      │
     SPG round 1│ LayoutEntry list                     │ NIXL WRITE (RDMA)
     SPG round 2│ agent metadata + descriptors         ▼
  ┌─────────────────────────── inference process ────────────────────────────┐
  │                                                                          │
  │  NIXLWeightUpdateWorker                                                  │
  │  ├─ NixlAgentWrapper                                                     │
  │  ├─ named_tensors               (vLLM params + scale buffers)            │
  │  └─ descriptors                 (serialized per-chunk xfer dlists)       │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
```

## Ownership

| Class | Scope | Responsibility |
|---|---|---|
| `ConversionSpec` | one logical parameter | How N trainer-side sources fuse + quantize into one vLLM destination. Model-agnostic. |
| `QuantizationSpec` | one destination | Dtype + optional FP8 scale suffix. `apply(src, out, sf)` writes in-place. |
| `PreTrainedModelPrimeRL.conversion_specs(i)` | one layer | Returns the `tuple[ConversionSpec, ...]` for layer `i`. Only model-specific hook. |
| `ShardedSlot` | one FSDP-sharded source | Holds this rank's shard. Writes to `chunk[my_rank]` on every inference peer. |
| `GatheredSlot` | one non-sharded source | Holds the full tensor. Writes once per peer, round-robin by `i % trainer_ws == my_rank`. |
| `ExpertSlot` | one fused expert param | Holds `num_local_experts` stacked weights. Writes per-(local, peer) filtered by peer's `expert_map`. |
| `TransportPlan` | one trainer process | Owns slots, NIXL registrations, rendezvous, write table, per-step push. Only class aware of FSDP/EP topology + trainer↔inference mapping. |
| `NIXLWeightBroadcast` | one trainer process | Thin orchestrator: creates the agent + SPG, delegates to the plan, runs the orchestrator handshake. |
| `NIXLWeightUpdateWorker` | one inference vLLM worker | Mirrors the layout: narrows vLLM tensors per `LayoutEntry`, publishes per-chunk descriptors, sits on SPG barrier during push. |

## Lifecycle — trainer side

```
TransportPlan(model, parallel_dims)        # construct
  └─ for each layer, for each ConversionSpec, spec.build_slots(...)
     └─ dispatch via spec.get_handler_class(src, parallel_dims)
        ShardedSlot | GatheredSlot | ExpertSlot
     → slots allocated inside classic_cuda_alloc() (NIXL-safe pool)

plan.register(agent)                        # phase 1
  └─ for each slot buffer: agent.register_tensor + agent.prep_local
     → self._local_preps[key] = prep handle

plan.rendezvous(spg, agent, inference_ws)   # phase 2
  ├─ SPG round 1: send list[LayoutEntry]
  ├─ SPG round 2: send agent_metadata, recv peer metadata + descriptors
  ├─ agent.add_remote + agent.make_connection per peer
  └─ _build_write_table(): for each slot.build_writes(peers) →
        _ResolvedWrite(local_prep, local_idx, remote_prep, peer, tag)

plan.push_once(model, agent, spg)           # phase 3, per training step
  ├─ state_dict = model.state_dict()        # DTensors (plus a few buffers)
  ├─ for each slot: slot.convert(state_dict)   # quantize / dtype-cast in-place
  ├─ torch.cuda.synchronize()
  ├─ for each _ResolvedWrite: agent.post_write(..., remote_idx=0)
  │     (remote_idx is always 0 — chunk selection already happened at prep time)
  ├─ drain every flush_every posts
  └─ spg.barrier()                           # cross-cluster ack
```

## Lifecycle — inference side

```
NIXLWeightUpdateWorker.init_nixl_transfer(...)
  ├─ agent.register_tensor(t) for every vLLM param + *_weight_scale_inv buffer
  ├─ SPG round 1: recv trainer's list[LayoutEntry], send expert_map
  ├─ build chunked descriptors:
  │     for each entry: narrow(inference_name, offset_rows, rows),
  │         split into num_chunks, publish one serialized dlist per chunk
  │     for each expert tensor: split into num_local chunks
  └─ SPG round 2: send agent_metadata + descriptors, recv trainer metadata

update_weights_from_path()                   # per training step
  ├─ spg.barrier()                           # wait for trainer drain + ack
  └─ update_mla_absorbed_weights(model)      # recompute MLA W_UV/W_UK_T
```

## Wire format

### Round 1 — trainer → inference

```python
{"role": "trainer", "global_rank": int, "layout_entries": list[LayoutEntry]}
```

One `LayoutEntry` per (slot, buffer) for non-expert slots:

```python
LayoutEntry(
    slot_key: str,         # descriptor publish key (trainer side uses same)
    inference_name: str,   # vLLM-side fused destination param name
    offset_rows: int,      # this source's row offset within the fused dst
    rows: int,             # this source's row count
    num_chunks: int,       # trainer_ws for per_shard, 1 for gather
)
```

Expert slots emit **zero** LayoutEntries — they route via `expert_map`
instead.

### Round 1 — inference → trainer

```python
{"role": "inference", "global_rank": int,
 "expert_map": {moe_prefix: list[global_expert_id]}}
```

### Round 2 — both directions

```python
{"role": ..., "global_rank": int,
 "agent_name": str, "agent_metadata": bytes,
 # inference side also:
 "descriptors": {slot_key: list[serialized_dlist_bytes]},
 "expert_map": {moe_prefix: list[global_expert_id]}}
```

Round 2 is deferred so each side's `get_metadata()` already reflects every
chunk registration from round-1-derived work.

### RDMA WRITE

Built once at rendezvous, reused every push. Each write targets one local
chunk and one remote chunk:

* `local_prep` — N-entry dlist from `agent.chunked_descs(tensor, num_chunks)`
  registered once at `register()`. `local_idx` picks which chunk.
* `remote_prep` — 1-entry dlist from `agent.prep_remote(peer,
  deserialize_descs(peer.descriptors[key][chunk_idx]))`. Chunk selection
  already happened at prep time, so `remote_idx` is always **0**. Preps are
  cached per `(peer, key, chunk_idx)` so repeat writes reuse them.

## Slot write patterns

| Slot | `num_chunks` | `local_idx` | `remote_chunk_idx` | Fan-out |
|---|---|---|---|---|
| `ShardedSlot` | 1 (weight holds only this rank's shard) | 0 | `my_rank` (writes into every peer's chunk for this rank) | all peers |
| `GatheredSlot` | 1 (full tensor) | 0 | 0 | `i % trainer_ws == my_rank` (round-robin across trainer ranks) |
| `ExpertSlot` | `num_local_experts` (one chunk per local expert) | `local_expert_idx` | `peer.expert_map[moe_prefix].index(global_id)` | peers owning the expert |

## Invariants

* **DTensor where possible.** Params are DTensors after `fully_shard`.
  `mlp.expert_bias` (and any other `register_buffer`) stays plain —
  `_resolve_source` guards with `isinstance(src, DTensor)` for that reason.
* **Classic cudaMalloc for slots.** The NIXL-registered buffers must not
  come from PyTorch's VMM expandable-segments pool — mlx5 returns "Local
  protection" on cuMemMap-backed VA. `TransportPlan.__init__` wraps slot
  allocation in `classic_cuda_alloc()`.
* **One MR per logical buffer.** Inference registers the full vLLM tensor
  once. Per-chunk xfer descriptors resolve to that MR's rkey at write
  time — overlapping per-chunk registrations confuse mlx5 and trip local
  protection errors.
* **Phase-gated plan.** `register()` must run before `rendezvous()`;
  `rendezvous()` before `push_once()`. Each slot captures `my_rank` /
  `trainer_ws` / `owned_global_experts` at construction — runtime methods
  only take dynamic inputs (state dict, peers, agent).
* **Flush or wedge.** NIXL/UCX's RC send queue wedges if too many WRITEs
  are posted before being drained. `push_once(flush_every=100)` drains
  every 100 posts.
