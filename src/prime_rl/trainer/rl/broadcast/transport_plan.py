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

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.slots import (
    ExpertSlot,
    LayoutEntry,
    PeerInfo,
    Slot,
)
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.classic_cuda_pool import classic_cuda_alloc
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nixl_transfer import NixlAgentWrapper


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

        # Populated in register().
        self._local_preps: dict[str, Any] = {}
        # Populated in rendezvous().
        self._peers: list[PeerInfo] = []
        self._writes: list[_ResolvedWrite] = []
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

        num_expert_slots = sum(1 for s in self.slots if isinstance(s, ExpertSlot))
        self.logger.info(
            f"NIXL transfer initialized: rank={self.my_rank} slots={len(self.slots)} "
            f"(expert={num_expert_slots}) writes={len(self._writes)} "
            f"bytes_per_push={self._bytes_per_push / 1e6:.2f} MB"
        )

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
        device = next(model.parameters()).device
        t_start = time.perf_counter()

        state_dict = model.state_dict()
        for slot in self.slots:
            slot.convert(state_dict)
        torch.cuda.synchronize(device)
        t_converted = time.perf_counter()

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
