"""TransportPlan: the model/parallel-aware glue between trainer-side slots
and the NIXL wire protocol.

A ``TransportPlan`` is constructed once per training process from a
``(model, parallel_dims)`` pair. It walks every layer's
:class:`ConversionSpec` table, builds the appropriate
:class:`~prime_rl.trainer.models.slots.Slot` instances inside the
classic-``cudaMalloc`` pool required by NIXL, and then moves through three
phase-gated stages:

1. :meth:`register` — pins every slot buffer for RDMA with a given
   :class:`NixlAgentWrapper` and records local prep handles.
2. :meth:`rendezvous` — runs the two SPG rounds with inference, parses the
   peer payloads into :class:`PeerInfo`, and materializes the write table
   from each slot's :meth:`build_writes`.
3. :meth:`push_once` — converts live model weights into the slot buffers
   and posts every RDMA WRITE, draining every ``flush_every`` posts to
   keep UCX's RC send queue bounded.

The plan is the only place aware of FSDP/EP topology and of the
trainer↔inference mapping; slots never see either beyond what they
capture at construction time.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
from prime_rl.trainer.models.slots import (
    ExpertSlot,
    GatheredSlot,
    LayoutEntry,
    PeerInfo,
    ShardedSlot,
    Slot,
)
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.classic_cuda_pool import classic_cuda_alloc
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nixl_transfer import NixlAgentWrapper


# Diagnostic: when set, push_once delegates to a "serial gather" path — rank 0
# gathers each slot's full unsharded tensor, quantizes into a shared scratchpad,
# and issues every RDMA WRITE itself (sequentially per slot, per layer). Used to
# isolate whether the default concurrent per-shard writer pattern contributes to
# residual KL drift.
_SERIAL_PUSH = os.environ.get("PRIME_RL_NIXL_SERIAL_PUSH") == "1"


@dataclass
class _ResolvedWrite:
    """A WriteEntry with NIXL prep handles resolved; ready for post_write."""

    local_prep: Any
    local_idx: int
    remote_prep: Any
    peer_name: str
    tag: str


class TransportPlan:
    """Owns the slots, the NIXL registrations, and the write table.

    Lifecycle:

    * ``plan = TransportPlan(model, parallel_dims)``
    * ``plan.register(agent)``
    * ``plan.rendezvous(spg, agent)``
    * ``plan.push_once(model, agent)`` per training step
    """

    def __init__(self, model: PreTrainedModelPrimeRL, parallel_dims: ParallelDims) -> None:
        self.logger = get_logger()
        self.parallel_dims = parallel_dims
        self.transfer_mode = os.environ.get("PRIME_RL_NIXL_TRANSFER_MODE", "all")
        # Per-replica NIXL coordinates. With HSDP, only replica-0 ranks ever
        # reach this constructor; ``my_rank`` indexes into the dp_shard_cp
        # axis (equivalently the SPG trainer-rank range), not the global
        # process rank.
        if dist.is_initialized() and parallel_dims.dp_replicate_enabled:
            shard_mesh = parallel_dims.get_mesh("dp_shard_cp")
            self.my_rank = shard_mesh.get_local_rank()
            self.trainer_ws = shard_mesh.size()
        else:
            self.my_rank = dist.get_rank() if dist.is_initialized() else 0
            self.trainer_ws = parallel_dims.world_size

        state_dict = model.state_dict()
        self.slots: list[Slot] = []
        # NIXL-registered slot buffers must be backed by classic cudaMalloc,
        # not PyTorch's VMM-backed expandable segments — the mlx5 HCA fails
        # translation on cuMemMap-backed VA and returns "Local protection".
        with classic_cuda_alloc():
            for layer_idx in range(model.config.num_hidden_layers):
                prefix = f"model.layers.{layer_idx}"
                for spec in model.conversion_specs(layer_idx):
                    self.slots.extend(spec.build_slots(prefix, state_dict, parallel_dims))
            # Non-layer tensors (e.g. embed_tokens, model.norm, lm_head).
            # Without these, inference never gets updates for top-level
            # params and KL drifts as trainer gradients advance them.
            for spec in model.non_layer_conversion_specs():
                self.slots.extend(spec.build_slots("", state_dict, parallel_dims))

        if self.transfer_mode == "non_expert_only":
            before = len(self.slots)
            self.slots = [slot for slot in self.slots if not isinstance(slot, ExpertSlot)]
            self.logger.info(
                "NIXL transfer mode non_expert_only: kept %d/%d slots",
                len(self.slots),
                before,
            )
        elif self.transfer_mode == "expert_only":
            before = len(self.slots)
            self.slots = [slot for slot in self.slots if isinstance(slot, ExpertSlot)]
            self.logger.info(
                "NIXL transfer mode expert_only: kept %d/%d slots",
                len(self.slots),
                before,
            )
        elif self.transfer_mode != "all":
            raise ValueError(
                f"Unsupported PRIME_RL_NIXL_TRANSFER_MODE={self.transfer_mode!r}; "
                "expected one of: all, non_expert_only, expert_only"
            )

        # Diagnostic: surface any state_dict keys we'd normally expect to
        # transport but none of the conversion specs cover. If a tensor is
        # in the trainer's live state_dict but NIXL skips it, inference
        # stays at its initial value while trainer drifts by gradient —
        # precisely the failure mode we're chasing.
        if self.my_rank == 0:
            tracked: set[str] = set()
            for slot in self.slots:
                if hasattr(slot, "source_name"):
                    tracked.add(slot.source_name)
                if hasattr(slot, "source_names"):
                    tracked.update(slot.source_names)
            # Only flag things that look like model parameters/buffers (not
            # optimizer state, step counters, etc.).
            relevant = {
                k for k in state_dict.keys()
                if (k.startswith("model.") or k == "lm_head.weight")
                and not k.startswith("model.layers.") == False  # keep both
            }
            # Simpler: everything model.*  or lm_head.weight.
            relevant = {k for k in state_dict.keys() if k.startswith("model.") or k == "lm_head.weight"}
            if self.transfer_mode == "non_expert_only":
                relevant = {k for k in relevant if ".mlp.experts." not in k}
            elif self.transfer_mode == "expert_only":
                relevant = {k for k in relevant if ".mlp.experts." in k}
            untracked = sorted(relevant - tracked)
            if untracked:
                self.logger.warning(
                    f"[nixl UNTRACKED] {len(untracked)} model state_dict keys not in any "
                    f"conversion spec (they will NOT be transported): first 30 = {untracked[:30]}"
                )
            else:
                self.logger.info(f"[nixl UNTRACKED] ok — every model state_dict key is tracked by some slot")

        # Populated in register().
        self._local_preps: dict[str, Any] = {}
        # Populated in rendezvous().
        self._peers: list[PeerInfo] = []
        self._writes: list[_ResolvedWrite] = []
        # Push counter for one-shot diagnostic dumps.
        self._push_counter = 0

        # Serial-push state (rank 0 only; populated in register + rendezvous).
        self._serial_enabled = _SERIAL_PUSH
        self._serial_slot_plans: dict[str, dict[str, Any]] = {}
        self._serial_scratch_w: Any = None
        self._serial_scratch_s: Any = None
        # {(slot_key_or_scale_key, remote_chunk_idx, peer_name): remote_prep}
        self._serial_remote_preps: dict[tuple[str, int, str], Any] = {}
        self._serial_model_ref = model
        if self._serial_enabled:
            self.logger.info(f"[nixl serial push] enabled on rank {self.my_rank}")
        # Derived statistic: total bytes moved per push (sum over all slot buffers,
        # weight + scale). Fan-out across inference peers is separate.
        self._bytes_per_push = sum(
            buf.numel() * buf.element_size() for slot in self.slots for _, buf, _ in slot.buffers
        )

    # ------------------------------------------------------------------ #
    # Phase 1: registration
    # ------------------------------------------------------------------ #

    def register(self, agent: NixlAgentWrapper) -> None:
        """Pin every slot buffer for RDMA and record local prep handles.

        Each slot exposes 1..N chunk counts per buffer via :attr:`buffers`.
        For non-expert slots num_chunks is always 1; for expert slots it is
        ``num_local_experts``, so NIXL can resolve per-expert sub-ranges
        via xfer descriptors at write time.
        """
        for slot in self.slots:
            for key, tensor, num_chunks in slot.buffers:
                agent.register_tensor(tensor)
                descs = agent.chunked_descs(tensor, num_chunks)
                self._local_preps[key] = agent.prep_local(descs)

        if self._serial_enabled and self.my_rank == 0:
            self._serial_allocate_and_register(agent)

    # ------------------------------------------------------------------ #
    # Serial-push helpers (rank 0 only)
    # ------------------------------------------------------------------ #

    def _serial_full_shape_for_slot(self, slot: Slot) -> tuple[int, ...]:
        """Full unsharded destination shape a slot would hold if rank 0 materialized it."""
        state_dict = self._serial_model_ref.state_dict()
        if isinstance(slot, ExpertSlot):
            src_shapes = [state_dict[name].shape for name in slot.source_names]
            dst = list(src_shapes[0])
            dst[slot.cat_dim] = sum(sh[slot.cat_dim] for sh in src_shapes)
            return tuple(dst)
        return tuple(state_dict[slot.source_name].shape)

    def _serial_allocate_and_register(self, agent: NixlAgentWrapper) -> None:
        """On rank 0: compute per-slot full shapes, allocate shared scratchpads
        sized to the largest slot's weight + scale, and pre-build per-chunk
        local preps into those scratchpads."""
        max_w_nbytes = 0
        max_s_nbytes = 0
        plans: dict[str, dict[str, Any]] = {}

        for slot in self.slots:
            full_shape = self._serial_full_shape_for_slot(slot)
            dtype = slot.weight.dtype
            elem = slot.weight.element_size()
            w_nbytes = 1
            for d in full_shape:
                w_nbytes *= d
            w_nbytes *= elem

            # Inference-side chunk count along dim 0:
            #   ExpertSlot   → one chunk per global expert
            #   ShardedSlot  → trainer_ws chunks of rows_per_shard rows
            #   GatheredSlot → single chunk covering the full tensor
            if isinstance(slot, ExpertSlot):
                num_chunks = full_shape[0]
            elif isinstance(slot, ShardedSlot):
                num_chunks = slot.trainer_ws
            else:
                num_chunks = 1
            w_chunk_nbytes = w_nbytes // num_chunks

            plan: dict[str, Any] = {
                "slot": slot,
                "full_shape": full_shape,
                "dtype": dtype,
                "w_nbytes": w_nbytes,
                "w_chunk_nbytes": w_chunk_nbytes,
                "num_chunks": num_chunks,
                "is_expert": isinstance(slot, ExpertSlot),
                "has_scale": slot.scale is not None,
            }
            if slot.scale is not None:
                if isinstance(slot, ExpertSlot):
                    scale_shape = (
                        full_shape[0],
                        ceil_div(full_shape[1], BLOCK_SIZE),
                        ceil_div(full_shape[2], BLOCK_SIZE),
                    )
                else:
                    scale_shape = (
                        ceil_div(full_shape[0], BLOCK_SIZE),
                        ceil_div(full_shape[1], BLOCK_SIZE),
                    )
                s_elem = slot.scale.element_size()
                s_nbytes = 1
                for d in scale_shape:
                    s_nbytes *= d
                s_nbytes *= s_elem
                s_chunk_nbytes = s_nbytes // num_chunks
                plan["scale_shape"] = scale_shape
                plan["scale_dtype"] = slot.scale.dtype
                plan["s_nbytes"] = s_nbytes
                plan["s_chunk_nbytes"] = s_chunk_nbytes
                max_s_nbytes = max(max_s_nbytes, s_nbytes)
            plans[slot.slot_key] = plan
            max_w_nbytes = max(max_w_nbytes, w_nbytes)

        self.logger.info(
            f"[nixl serial push] allocating scratchpads: weight={max_w_nbytes / 1e9:.2f} GB "
            f"scale={max_s_nbytes / 1e6:.2f} MB over {len(plans)} slots"
        )
        device = torch.device("cuda", torch.cuda.current_device())
        with classic_cuda_alloc():
            self._serial_scratch_w = torch.empty(max_w_nbytes, dtype=torch.uint8, device=device)
            if max_s_nbytes > 0:
                self._serial_scratch_s = torch.empty(max_s_nbytes, dtype=torch.uint8, device=device)
        agent.register_tensor(self._serial_scratch_w)
        if self._serial_scratch_s is not None:
            agent.register_tensor(self._serial_scratch_s)

        # Pre-build per-chunk local preps for every slot. Scratchpads are
        # reused across slots: plan's offsets are always 0..nbytes inside
        # the scratchpad, and we drain between slots.
        for slot_key, plan in plans.items():
            w_base = self._serial_scratch_w.data_ptr()
            dev_idx = device.index
            w_preps: list[Any] = []
            for c in range(plan["num_chunks"]):
                descs = agent.descs_from_tuples([(w_base + c * plan["w_chunk_nbytes"], plan["w_chunk_nbytes"], dev_idx)])
                w_preps.append(agent.prep_local(descs))
            plan["w_local_preps"] = w_preps
            if plan.get("has_scale"):
                s_base = self._serial_scratch_s.data_ptr()
                s_preps: list[Any] = []
                for c in range(plan["num_chunks"]):
                    descs = agent.descs_from_tuples([
                        (s_base + c * plan["s_chunk_nbytes"], plan["s_chunk_nbytes"], dev_idx)
                    ])
                    s_preps.append(agent.prep_local(descs))
                plan["s_local_preps"] = s_preps
        self._serial_slot_plans = plans

    # ------------------------------------------------------------------ #
    # Phase 2: rendezvous
    # ------------------------------------------------------------------ #

    def rendezvous(self, spg: StatelessProcessGroup, agent: NixlAgentWrapper, inference_ws: int) -> None:
        """Run the two SPG rounds and build the write table.

        Round 1 is layout-only: trainer ships a flat :class:`LayoutEntry`
        list so inference can narrow its vLLM tensors and build per-chunk
        descriptors. Agent metadata is deferred to round 2 so the metadata
        each side sees already covers every chunk registration from
        :meth:`register`.
        """
        layout_entries: list[LayoutEntry] = []
        for slot in self.slots:
            layout_entries.extend(slot.layout_payload())
        spg.all_gather_obj(
            {"role": "trainer", "global_rank": self.my_rank, "layout_entries": layout_entries}
        )

        round2 = spg.all_gather_obj(
            {
                "role": "trainer",
                "global_rank": self.my_rank,
                "agent_name": agent.name,
                "agent_metadata": agent.get_metadata(),
            }
        )
        inference_infos = round2[self.trainer_ws : self.trainer_ws + inference_ws]
        for peer in inference_infos:
            agent.add_remote(peer["agent_metadata"])
            agent.make_connection(peer["agent_name"])
        self._peers = [
            PeerInfo(
                agent_name=peer["agent_name"],
                descriptors=peer["descriptors"],
                expert_map=peer["expert_map"],
            )
            for peer in inference_infos
        ]
        self._build_write_table(agent)

        if self._serial_enabled and self.my_rank == 0:
            self._serial_build_remote_preps(agent)

        num_expert_slots = sum(1 for s in self.slots if isinstance(s, ExpertSlot))
        self.logger.info(
            f"NIXL transfer initialized: rank={self.my_rank} slots={len(self.slots)} "
            f"(expert={num_expert_slots}) writes={len(self._writes)} "
            f"bytes_per_push={self._bytes_per_push / 1e6:.2f} MB"
        )

    def _serial_build_remote_preps(self, agent: NixlAgentWrapper) -> None:
        """Rank-0-only: cache remote preps for every peer × slot × chunk combo."""
        n = 0
        for peer in self._peers:
            for key, serialized_list in peer.descriptors.items():
                for chunk_idx, serialized in enumerate(serialized_list):
                    dlist = agent.deserialize_descs(serialized)
                    prep = agent.prep_remote(peer.agent_name, dlist)
                    self._serial_remote_preps[(key, chunk_idx, peer.agent_name)] = prep
                    n += 1
        self.logger.info(f"[nixl serial push] cached {n} remote preps across {len(self._peers)} peers")

    def _build_write_table(self, agent: NixlAgentWrapper) -> None:
        """Resolve every slot's WriteEntry list into cached NIXL prep handles."""
        remote_cache: dict[tuple[str, str, int], Any] = {}

        def _resolve_remote(peer: PeerInfo, remote_key: str, chunk_idx: int) -> Any:
            cache_key = (peer.agent_name, remote_key, chunk_idx)
            cached = remote_cache.get(cache_key)
            if cached is not None:
                return cached
            remote_dlist = agent.deserialize_descs(peer.descriptors[remote_key][chunk_idx])
            cached = agent.prep_remote(peer.agent_name, remote_dlist)
            remote_cache[cache_key] = cached
            return cached

        peers_by_name = {p.agent_name: p for p in self._peers}
        writes: list[_ResolvedWrite] = []
        for slot in self.slots:
            for w in slot.build_writes(self._peers):
                peer = peers_by_name[w.peer_name]
                remote_prep = _resolve_remote(peer, w.remote_buffer_key, w.remote_chunk_idx)
                writes.append(
                    _ResolvedWrite(
                        # remote_prep was built from exactly one serialized
                        # dlist entry (see _resolve_remote) so post_write's
                        # index into it is always 0 — chunk selection
                        # already happened at prep time.
                        local_prep=self._local_preps[w.local_buffer_key],
                        local_idx=w.local_chunk_idx,
                        remote_prep=remote_prep,
                        peer_name=w.peer_name,
                        tag=w.tag,
                    )
                )
        self._writes = writes

    # ------------------------------------------------------------------ #
    # Diagnostic dump (one-shot) — byte-level trainer/inference diff
    # ------------------------------------------------------------------ #

    # Scope of slots to dump. Layer 3 covers all slot types once; non-layer
    # covers embed/norm/lm_head. Broad enough to catch any routing bug.
    _DUMP_LAYER_PREFIX = "model.layers.3."
    _DUMP_NON_LAYER = frozenset({
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    })

    def _maybe_dump_trainer(self) -> None:
        """Save per-rank slot contents to ``$NIXL_DUMP_DIR/trainer_r{RRR}.pt``
        when this push's counter matches ``$NIXL_DUMP_PUSH``. Pair with the
        inference-side dump and the tools/nixl_diff.py script for a
        byte-exact cross-check."""
        dump_dir = os.environ.get("NIXL_DUMP_DIR")
        target_push = int(os.environ.get("NIXL_DUMP_PUSH", "0"))
        if not dump_dir or self._push_counter != target_push:
            return
        # Rank 0 creates the dir; other ranks will write files shortly after
        # once their convert completes.
        if self.my_rank == 0:
            os.makedirs(dump_dir, exist_ok=True)

        out: dict[str, dict[str, Any]] = {}
        for slot in self.slots:
            k = slot.slot_key
            if not (k.startswith(self._DUMP_LAYER_PREFIX) or k in self._DUMP_NON_LAYER):
                continue
            entry: dict[str, Any] = {
                "type": type(slot).__name__,
                "slot_key": slot.slot_key,
                "weight": slot.weight.detach().to("cpu").clone(),
                "scale": slot.scale.detach().to("cpu").clone() if slot.scale is not None else None,
            }
            for attr in (
                "source_name", "source_names", "inference_name", "inference_scale_name",
                "offset_rows", "scale_offset_rows", "rows", "scale_rows",
                "my_rank", "trainer_ws", "moe_prefix", "owned_global_experts",
                "cat_dim", "scale_key",
            ):
                if hasattr(slot, attr):
                    v = getattr(slot, attr)
                    if isinstance(v, tuple):
                        v = list(v)
                    entry[attr] = v
            out[k] = entry

        fname = f"{dump_dir}/trainer_r{self.my_rank:03d}.pt"
        torch.save(out, fname)
        self.logger.info(f"[nixl DUMP] rank={self.my_rank} wrote {len(out)} slots to {fname}")

    # ------------------------------------------------------------------ #
    # Phase 3: per-step push
    # ------------------------------------------------------------------ #

    def push_once(
        self,
        model: PreTrainedModelPrimeRL,
        agent: NixlAgentWrapper,
        spg: StatelessProcessGroup,
        flush_every: int = 100,
    ) -> None:
        """Convert live weights into slot buffers and post all WRITEs.

        Draining every ``flush_every`` posts keeps NIXL/UCX's RC send queue
        bounded — posting all writes at once wedged the UCX progress engine
        on large world sizes. Ends with an SPG barrier so trainer and
        inference know every WRITE has been acknowledged before the
        orchestrator calls ``/resume``.
        """
        if self._serial_enabled:
            return self.push_once_serial(model, agent, spg)
        device = next(model.parameters()).device
        t_start = time.perf_counter()
        self._push_counter += 1

        state_dict = model.state_dict()
        for slot in self.slots:
            slot.convert(state_dict)
        torch.cuda.synchronize(device)
        t_converted = time.perf_counter()

        # Diagnostic one-shot dump of all slot contents (scope-limited).
        self._maybe_dump_trainer()

        # Diagnostic: rank-0 logs post-convert signatures for anchor
        # slots covering each slot type. Trainer's slot.weight holds
        # the exact content that's about to be RDMA-written. Inference
        # logs the matching param/buffer sum after the barrier. If
        # sums diverge for a given anchor, that slot type has a
        # transport or quantize bug.
        #   G (bf16 gather)   : input_layernorm.weight
        #   F (fp8 gather)    : self_attn.kv_b_proj.weight + scale
        #   E (fp8 expert)    : mlp.experts.w13_weight expert[0]
        if self.my_rank == 0:
            for slot in self.slots:
                # F_q: q_a_proj source (first source of fused_qkv_a_proj).
                #   inference fused_qkv_a_proj[0:2048] should == this.
                # F_kv: kv_a_proj_with_mqa source (second source of fused_qkv_a_proj).
                #   inference fused_qkv_a_proj[2048:2624] should == this.
                # If any region sum diverges, offset/routing is off for fused specs.
                if slot.slot_key == "model.layers.3.self_attn.q_a_proj.weight":
                    w = slot.weight
                    sc = slot.scale
                    w_bytes = w.view(torch.uint8).to(torch.int64).sum().item()
                    s_sum = sc.to(torch.float64).sum().item() if sc is not None else 0.0
                    self.logger.info(
                        f"[nixl SIG trainer] anchor=F_q key={slot.slot_key} "
                        f"w_bytes={w_bytes} w_shape={tuple(w.shape)} "
                        f"scale={s_sum:.8f}"
                    )
                elif slot.slot_key == "model.layers.3.self_attn.kv_a_proj_with_mqa.weight":
                    w = slot.weight
                    sc = slot.scale
                    w_bytes = w.view(torch.uint8).to(torch.int64).sum().item()
                    s_sum = sc.to(torch.float64).sum().item() if sc is not None else 0.0
                    self.logger.info(
                        f"[nixl SIG trainer] anchor=F_kv key={slot.slot_key} "
                        f"w_bytes={w_bytes} w_shape={tuple(w.shape)} "
                        f"scale={s_sum:.8f}"
                    )
                elif slot.slot_key in (
                    "model.embed_tokens.weight",
                    "model.norm.weight",
                    "lm_head.weight",
                ):
                    # N anchors for non-layer tensors added in iter8.
                    w = slot.weight
                    self.logger.info(
                        f"[nixl SIG trainer] anchor=N key={slot.slot_key} "
                        f"sum={w.to(torch.float64).sum().item():.8f} "
                        f"shape={tuple(w.shape)}"
                    )
                elif slot.slot_key == "model.layers.3.mlp.experts.w13_weight":
                    # E anchors for 3 experts spread across the local block.
                    for local_idx in (0, 1, 2, 3):
                        if local_idx < slot.weight.shape[0]:
                            w = slot.weight[local_idx]
                            sc = slot.scale[local_idx] if slot.scale is not None else None
                            w_bytes = w.view(torch.uint8).to(torch.int64).sum().item()
                            s_sum = sc.to(torch.float64).sum().item() if sc is not None else 0.0
                            self.logger.info(
                                f"[nixl SIG trainer] anchor=E[E{local_idx}] "
                                f"key={slot.slot_key} "
                                f"w_bytes={w_bytes} w_shape={tuple(w.shape)} "
                                f"scale={s_sum:.8f}"
                            )

        handles: list = []
        handle_ctx: list[tuple[str, str]] = []

        def _drain() -> None:
            for h, ctx in zip(handles, handle_ctx):
                agent.wait(h, context=f"rank={self.my_rank} peer={ctx[0]} tag={ctx[1]}")
            handles.clear()
            handle_ctx.clear()

        # PRE-WRITE barrier: ensure every inference worker has actually entered
        # update_weights_from_path and is quiescent before we start RDMA-writing
        # into their weight memory. The orchestrator's /pause should do this
        # too, but /pause may return before all in-flight CUDA work actually
        # drains on inference — this is the belt-and-suspenders check.
        spg.barrier()

        for i, w in enumerate(self._writes):
            handles.append(agent.post_write(w.local_prep, w.local_idx, w.remote_prep, 0))
            handle_ctx.append((w.peer_name, w.tag))
            if (i + 1) % flush_every == 0:
                _drain()
        if handles:
            _drain()

        t_waited = time.perf_counter()
        spg.barrier()
        t_done = time.perf_counter()

        dt_convert = t_converted - t_start
        dt_post_wait = t_waited - t_converted
        dt_barrier = t_done - t_waited
        dt_total = t_done - t_start
        dt_wire = t_waited - t_start
        gbps_wire = self._bytes_per_push / dt_wire / 1e9 if dt_wire > 0 else 0.0
        gbps_net = self._bytes_per_push / dt_post_wait / 1e9 if dt_post_wait > 0 else 0.0
        self.logger.info(
            f"[nixl rank={self.my_rank}] push "
            f"bytes={self._bytes_per_push / 1e6:.2f}MB handles={len(self._writes)} "
            f"convert={dt_convert * 1e3:.2f}ms post+wait={dt_post_wait * 1e3:.2f}ms "
            f"barrier={dt_barrier * 1e3:.2f}ms total={dt_total * 1e3:.2f}ms "
            f"wire_bw={gbps_wire:.2f}GB/s net_bw={gbps_net:.2f}GB/s"
        )

    # ------------------------------------------------------------------ #
    # Serial-gather push path (diagnostic)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def push_once_serial(
        self,
        model: PreTrainedModelPrimeRL,
        agent: NixlAgentWrapper,
        spg: StatelessProcessGroup,
    ) -> None:
        """Gather full tensor to rank 0, quantize into scratchpad, write one slot at a time.

        Diagnostic alternative to :meth:`push_once` — isolates whether the
        default concurrent per-shard writer pattern contributes to residual
        KL drift. Semantics:

          1. Per layer (then non-layer tensors), per slot: every trainer rank
             calls ``full_tensor()`` on the slot's source(s) so the DTensor
             collective materializes the unsharded data everywhere. Non-rank-0
             ranks then drop the result.
          2. Rank 0 cats fused sources along ``cat_dim`` and runs the spec's
             quantization into a pre-registered scratchpad.
          3. Rank 0 posts one RDMA WRITE per (chunk, peer), expert-aware via
             ``peer.expert_map`` for :class:`ExpertSlot`. Drains before the
             next slot so the scratchpad can be reused.
          4. Bracketed by ``spg.barrier()`` so inference sees the same
             pause/resume contract as the default path.
        """
        t_start = time.perf_counter()
        self._push_counter += 1
        state_dict = model.state_dict()

        # Group slots by layer_idx (and non-layer bucket) for deterministic ordering.
        num_layers = model.config.num_hidden_layers
        per_layer: list[list[Slot]] = [[] for _ in range(num_layers)]
        non_layer: list[Slot] = []
        for slot in self.slots:
            k = slot.slot_key
            if k.startswith("model.layers."):
                per_layer[int(k.split(".")[2])].append(slot)
            else:
                non_layer.append(slot)

        # PRE-WRITE barrier (matches default push_once).
        spg.barrier()

        for layer_idx in range(num_layers):
            for slot in per_layer[layer_idx]:
                self._serial_push_one_slot(slot, state_dict, agent)
        for slot in non_layer:
            self._serial_push_one_slot(slot, state_dict, agent)

        if self.my_rank == 0:
            torch.cuda.synchronize()
        spg.barrier()
        t_done = time.perf_counter()
        self.logger.info(
            f"[nixl rank={self.my_rank}] serial-push "
            f"slots={len(self.slots)} total={(t_done - t_start) * 1e3:.2f}ms"
        )

    def _serial_push_one_slot(
        self,
        slot: Slot,
        state_dict: dict[str, Any],
        agent: NixlAgentWrapper,
    ) -> None:
        """One full gather + quantize + write-to-all-peers for a single slot.

        Every rank participates in ``full_tensor()``; only rank 0 quantizes and
        writes. Drains before returning so the scratchpad can be reused.
        """
        # 1. Materialize full tensor(s) on every rank (collective).
        if isinstance(slot, ExpertSlot):
            names = list(slot.source_names)
        else:
            names = [slot.source_name]
        full_srcs = [state_dict[n].full_tensor() for n in names]

        if self.my_rank != 0:
            # Non-writer ranks participated in the all_gather; release refs.
            del full_srcs
            return

        # 2. Cat if fused, quantize into scratchpad views.
        if len(full_srcs) == 1:
            src = full_srcs[0]
        else:
            cat_dim = slot.cat_dim if isinstance(slot, ExpertSlot) else slot.spec.cat_dim
            src = torch.cat(full_srcs, dim=cat_dim)

        plan = self._serial_slot_plans[slot.slot_key]
        w_view = (
            self._serial_scratch_w[: plan["w_nbytes"]]
            .view(plan["dtype"])
            .view(plan["full_shape"])
        )
        s_view = None
        if plan.get("has_scale"):
            assert self._serial_scratch_s is not None
            s_view = (
                self._serial_scratch_s[: plan["s_nbytes"]]
                .view(plan["scale_dtype"])
                .view(plan["scale_shape"])
            )
        slot.spec.quantization.apply(src, w_view, s_view)
        del src, full_srcs

        # 3. Post writes. For experts, write per-global-expert to every peer
        # that owns that expert. For non-experts, write every chunk to every
        # peer (full tensor redundantly landed at each peer).
        handles: list = []
        ctx: list[tuple[str, str]] = []

        def _drain() -> None:
            for h, c in zip(handles, ctx):
                agent.wait(h, context=f"rank=0 peer={c[0]} tag={c[1]}")
            handles.clear()
            ctx.clear()

        if isinstance(slot, ExpertSlot):
            num_global = plan["num_chunks"]
            for g in range(num_global):
                for peer in self._peers:
                    peer_experts = peer.expert_map.get(slot.moe_prefix, [])
                    if g not in peer_experts:
                        continue
                    remote_idx = peer_experts.index(g)
                    handles.append(agent.post_write(
                        plan["w_local_preps"][g], 0,
                        self._serial_remote_preps[(slot.slot_key, remote_idx, peer.agent_name)], 0,
                    ))
                    ctx.append((peer.agent_name, f"expert:{slot.slot_key}:E{g}"))
                    if plan.get("has_scale"):
                        handles.append(agent.post_write(
                            plan["s_local_preps"][g], 0,
                            self._serial_remote_preps[(slot.scale_key, remote_idx, peer.agent_name)], 0,
                        ))
                        ctx.append((peer.agent_name, f"expert:{slot.scale_key}:E{g}"))
        else:
            num_chunks = plan["num_chunks"]
            for c_idx in range(num_chunks):
                for peer in self._peers:
                    handles.append(agent.post_write(
                        plan["w_local_preps"][c_idx], 0,
                        self._serial_remote_preps[(slot.slot_key, c_idx, peer.agent_name)], 0,
                    ))
                    ctx.append((peer.agent_name, f"chunk:{slot.slot_key}:{c_idx}"))
                    if plan.get("has_scale"):
                        handles.append(agent.post_write(
                            plan["s_local_preps"][c_idx], 0,
                            self._serial_remote_preps[(slot.scale_key, c_idx, peer.agent_name)], 0,
                        ))
                        ctx.append((peer.agent_name, f"chunk:{slot.scale_key}:{c_idx}"))
        _drain()
