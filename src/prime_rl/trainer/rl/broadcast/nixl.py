"""Trainer-side NIXL (UCX/RDMA) weight sender.

Replaces the per-step ``.full_tensor()``/NCCL-broadcast path for GLM MoE DSA:
each trainer rank pushes its FSDP/EP shards directly into pre-registered
parameter memory on the inference side.

Routing is per-slot:
  * Expert tensors — fused, EP+FSDP-local. Routed per-expert via the inference
    side's ``expert_map`` (only peers that own a given global expert receive
    writes for it).
  * Non-expert tensors — one slot per source, replicated on every inference
    rank. ``per_shard`` slots hold the rank-local FSDP slice and are written
    to ``chunk[fsdp_rank]`` on **every** inference peer. ``gather`` slots hold
    the full tensor (via ``full_tensor()`` in the conversion pass) and are
    written once, round-robin to the inference rank matched by ``i % R``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nixl_transfer import NixlAgentWrapper, NixlTransferMeta, make_agent_name
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path

NIXL_READY_MARKER = "NIXL_READY"


def _is_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name


def create_nixl_metadata(model: PreTrainedModelPrimeRL, parallel_dims: ParallelDims) -> NixlTransferMeta:
    """Build :class:`NixlTransferMeta` from model + parallel setup."""
    if parallel_dims.ep_enabled:
        ep_mesh = parallel_dims.get_mesh("ep")
        ep_size, ep_rank = ep_mesh.size(), ep_mesh.get_local_rank()
        fsdp_mesh = parallel_dims.get_mesh("dp_shard_mod_ep")
        fsdp_size, fsdp_rank = fsdp_mesh.size(), fsdp_mesh.get_local_rank()
    else:
        ep_size, ep_rank = 1, 0
        fsdp_size, fsdp_rank = 1, 0

    num_experts_per_ep = model.config.n_routed_experts // ep_size
    num_local_experts = num_experts_per_ep // fsdp_size
    base = ep_rank * num_experts_per_ep + fsdp_rank * num_local_experts
    owned_global_experts = list(range(base, base + num_local_experts))
    return NixlTransferMeta(
        slots=model.allocate_slots(parallel_dims),
        num_layers=model.config.num_hidden_layers,
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_local_experts=num_local_experts,
        owned_global_experts=owned_global_experts,
        fsdp_size=fsdp_size,
        fsdp_rank=fsdp_rank,
        non_expert_layout=model.non_expert_slot_layout(parallel_dims),
    )


class NIXLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via NIXL (zero-copy RDMA)."""

    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        meta: NixlTransferMeta,
    ) -> None:
        super().__init__(output_dir)
        self.logger = get_logger()
        self.config = config
        self.world = get_world()
        self._multi_run_manager = None
        self._meta = meta

        self.logger.info(f"[nixl trainer rank={self.world.rank}] creating NIXL agent")
        self._agent = NixlAgentWrapper(
            name=make_agent_name("trainer", self.world.rank),
            local_rank=self.world.local_rank,
            backends=config.backends,
        )
        self.logger.info(f"[nixl trainer rank={self.world.rank}] agent created; registering slots")

        # Register each slot's full memory region once (one NIXL reg entry per
        # slot). Chunks are addressed via sub-range xfer descriptors at write
        # time — NIXL resolves the rkey from the containing registered region.
        local_preps: dict[str, Any] = {}
        n_slots = 0
        for layer_idx in range(meta.num_layers):
            for name, tensor in meta.slots[layer_idx].items():
                self._agent.register_tensor(tensor)
                if _is_expert_tensor(name):
                    descs = self._agent.chunked_descs(tensor, meta.num_local_experts)
                else:
                    descs = self._agent.chunked_descs(tensor, 1)
                local_preps[name] = self._agent.prep_local(descs)
                n_slots += 1
        self.logger.info(
            f"[nixl trainer rank={self.world.rank}] registered {n_slots} slots; "
            f"creating StatelessProcessGroup host={config.host} port={config.port} "
            f"rank={self.world.rank} world_size={self.world.world_size + config.inference_world_size}"
        )

        self._spg = StatelessProcessGroup.create(
            host=config.host,
            port=config.port,
            rank=self.world.rank,
            world_size=self.world.world_size + config.inference_world_size,
            store_timeout=config.timeout,
        )
        self.logger.info(f"[nixl trainer rank={self.world.rank}] SPG created; round 1 all_gather_obj")

        # Round 1 — share the layout only. Inference uses it to size its chunked
        # registrations; we wait until round 2 to exchange agent metadata so the
        # peer's metadata includes every chunk registration.
        self._spg.all_gather_obj(
            {
                "role": "trainer",
                "global_rank": self.world.rank,
                "non_expert_layout": meta.non_expert_layout,
            }
        )
        self.logger.info(
            f"[nixl trainer rank={self.world.rank}] round 1 done; round 2 all_gather_obj for descriptors"
        )

        # Round 2 — fresh agent_metadata (after all chunk registrations land on
        # both sides) + inference's serialized chunk descriptors.
        round2 = self._spg.all_gather_obj(
            {
                "role": "trainer",
                "global_rank": self.world.rank,
                "agent_name": self._agent.name,
                "agent_metadata": self._agent.get_metadata(),
            }
        )
        inference_infos = round2[self.world.world_size :]
        for peer in inference_infos:
            self._agent.add_remote(peer["agent_metadata"])
            self._agent.make_connection(peer["agent_name"])
        self.logger.info(f"[nixl trainer rank={self.world.rank}] round 2 done; building write table")

        # Build the write table. Each entry: (local_prep, local_idx, remote_prep,
        # remote_idx, peer_name, tag). Inference publishes a list of per-chunk
        # 1-entry dlists per slot; we deserialize the one matching our chunk and
        # prep it per-peer so make_prepped_xfer pairs 1-entry local ↔ 1-entry remote.
        trainer_ws = self.world.world_size
        my_rank = self.world.rank
        self._writes: list[tuple[Any, int, Any, int, str, str]] = []
        remote_prep_cache: dict[tuple[str, str, int], Any] = {}
        # Diagnostic: collect slot name → (local_bytes, remote_chunk_bytes) for
        # any non-expert per_shard slot so we can catch length mismatches at
        # init (the writer-side ``makeXferReq`` only surfaces them as INVALID_PARAM).
        first_size_logged = False

        def _get_remote_prep(peer_name: str, slot_name: str, chunk_idx: int, peer_descs: list[bytes]) -> Any:
            key = (peer_name, slot_name, chunk_idx)
            cached = remote_prep_cache.get(key)
            if cached is not None:
                return cached
            remote_dlist = self._agent.deserialize_descs(peer_descs[chunk_idx])
            cached = self._agent.prep_remote(peer_name, remote_dlist)
            remote_prep_cache[key] = cached
            return cached

        # Verify local/remote desc sizes match as we build the table — NIXL's
        # ``makeXferReq`` only surfaces mismatches at post time as INVALID_PARAM.
        import pickle as _pkl

        def _dlist_byte_size(bytes_blob: bytes) -> int:
            # Deserialized xfer dlist's descCount is 1 (single-entry); read size
            # from the underlying pickle structure for diagnostics only.
            try:
                obj = _pkl.loads(bytes_blob)
                return int(obj[0][1]) if hasattr(obj, "__getitem__") else -1
            except Exception:
                return -1

        def _check(tag: str, local_b: int, remote_b: int) -> None:
            if local_b != remote_b:
                self.logger.error(
                    f"[nixl rank={self.world.rank}] SIZE MISMATCH {tag} local={local_b} remote={remote_b}"
                )

        for layer_idx in range(meta.num_layers):
            for name, slot in meta.slots[layer_idx].items():
                local_prep = local_preps[name]
                slot_bytes = slot.numel() * slot.element_size()

                if _is_expert_tensor(name):
                    moe_prefix = f"model.layers.{layer_idx}.mlp.experts"
                    expert_bytes = slot_bytes // meta.num_local_experts
                    for local_idx, global_idx in enumerate(meta.owned_global_experts):
                        for peer in inference_infos:
                            if global_idx not in peer["expert_map"][moe_prefix]:
                                continue
                            remote_idx = peer["expert_map"][moe_prefix].index(global_idx)
                            _check(
                                f"expert:{name}:E{global_idx}@{peer['agent_name']}",
                                expert_bytes,
                                _dlist_byte_size(peer["descriptors"][name][remote_idx]),
                            )
                            remote_prep = _get_remote_prep(peer["agent_name"], name, remote_idx, peer["descriptors"][name])
                            self._writes.append(
                                (local_prep, local_idx, remote_prep, 0, peer["agent_name"], f"expert:{name}:E{global_idx}")
                            )
                    continue

                info = meta.non_expert_layout[layer_idx][name]
                if info["handling"] == "per_shard":
                    for peer in inference_infos:
                        _check(
                            f"per_shard:{name}@{peer['agent_name']}",
                            slot_bytes,
                            _dlist_byte_size(peer["descriptors"][name][my_rank]),
                        )
                        remote_prep = _get_remote_prep(peer["agent_name"], name, my_rank, peer["descriptors"][name])
                        self._writes.append(
                            (local_prep, 0, remote_prep, 0, peer["agent_name"], f"per_shard:{name}")
                        )
                else:
                    for i, peer in enumerate(inference_infos):
                        if i % trainer_ws != my_rank:
                            continue
                        _check(
                            f"gather:{name}@{peer['agent_name']}",
                            slot_bytes,
                            _dlist_byte_size(peer["descriptors"][name][0]),
                        )
                        remote_prep = _get_remote_prep(peer["agent_name"], name, 0, peer["descriptors"][name])
                        self._writes.append(
                            (local_prep, 0, remote_prep, 0, peer["agent_name"], f"gather:{name}")
                        )

        self._bytes_per_push = sum(
            t.numel() * t.element_size()
            for slots in meta.slots.values()
            for t in slots.values()
        )

        self.logger.info(
            f"NIXL transfer initialized: rank={self.world.rank} "
            f"owned_experts={meta.owned_global_experts} writes={len(self._writes)} "
            f"bytes_per_push={self._bytes_per_push / 1e6:.2f} MB"
        )

    @torch.no_grad()
    def push_once(self, model: PreTrainedModelPrimeRL) -> None:
        """Convert every layer into its stable slot and post writes in chunks.

        Draining handles every ``flush_every`` posts keeps NIXL/UCX queue
        depth bounded. Posting all 28k handles at once appeared to overwhelm
        the UCX progress engine — completions came back so slowly that wall
        time dwarfed the useful bytes/sec. Progress logs every chunk show
        where we are if the run ever stalls.
        """
        device = next(model.parameters()).device

        t_start = time.perf_counter()

        for layer_idx in range(self._meta.num_layers):
            model.convert_layer_to_vllm_kernel(layer_idx, out_buffers=self._meta.slots[layer_idx])
        torch.cuda.synchronize(device)
        t_converted = time.perf_counter()

        flush_every = 100
        handles: list = []
        handle_ctx: list[tuple[str, str]] = []  # parallel list of (peer, tag) for error context

        def _drain(at: int) -> None:
            t_drain = time.perf_counter()
            for h, ctx in zip(handles, handle_ctx):
                self._agent.wait(h, context=f"rank={self.world.rank} peer={ctx[0]} tag={ctx[1]}")
            handles.clear()
            handle_ctx.clear()
            self.logger.info(
                f"[nixl rank={self.world.rank}] drained through {at}/{len(self._writes)} in {(time.perf_counter() - t_drain) * 1e3:.1f}ms"
            )

        for i, (lp, li, rp, ri, peer_name, tag) in enumerate(self._writes):
            handles.append(self._agent.post_write(lp, li, rp, ri))
            handle_ctx.append((peer_name, tag))
            if (i + 1) % flush_every == 0:
                _drain(i + 1)
        if handles:
            _drain(len(self._writes))

        t_posted = time.perf_counter()
        t_waited = t_posted

        self.logger.info(f"[nixl rank={self.world.rank}] entering SPG barrier after drain")
        self._spg.barrier()
        t_done = time.perf_counter()
        self.logger.info(f"[nixl rank={self.world.rank}] left SPG barrier in {(t_done - t_waited) * 1e3:.1f}ms")

        dt_convert = t_converted - t_start
        dt_post = t_posted - t_converted
        dt_wait = t_waited - t_posted
        dt_barrier = t_done - t_waited
        dt_total = t_done - t_start
        dt_wire = t_waited - t_start
        gbps_wire = self._bytes_per_push / dt_wire / 1e9 if dt_wire > 0 else 0.0
        gbps_net = self._bytes_per_push / (dt_post + dt_wait) / 1e9 if (dt_post + dt_wait) > 0 else 0.0

        self.logger.info(
            f"[nixl rank={self.world.rank}] push "
            f"bytes={self._bytes_per_push / 1e6:.2f}MB handles={len(handles)} "
            f"convert={dt_convert * 1e3:.2f}ms post={dt_post * 1e3:.2f}ms "
            f"wait={dt_wait * 1e3:.2f}ms barrier={dt_barrier * 1e3:.2f}ms "
            f"total={dt_total * 1e3:.2f}ms wire_bw={gbps_wire:.2f}GB/s net_bw={gbps_net:.2f}GB/s"
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        start = time.perf_counter()
        notified_runs: list[tuple[int, Path]] = []
        if self.world.is_master:
            notified_runs = self._notify_orchestrator()
            self._wait_for_nixl_ready(notified_runs)
        # Only rank 0 waits for NIXL_READY (the orchestrator drops it after all
        # inference engines acknowledge /pause). Hold every other trainer rank
        # here so no rank starts RDMA-writing into inference memory before all
        # engines have drained — otherwise a worker still running a forward pass
        # races the write into its own weight buffers.
        dist.barrier()
        self.push_once(model)
        self.logger.debug(f"NIXL weights broadcasted in {time.perf_counter() - start:.2f}s")

    @property
    def multi_run_manager(self):
        if self._multi_run_manager is None:
            self._multi_run_manager = get_multi_run_manager()
        return self._multi_run_manager

    def _notify_orchestrator(self) -> list[tuple[int, Path]]:
        notified_runs: list[tuple[int, Path]] = []
        for idx in self.multi_run_manager.used_idxs:
            if not self.multi_run_manager.ready_to_update[idx]:
                continue
            save_dir = get_step_path(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                self.multi_run_manager.progress[idx].step,
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "STABLE").touch()
            notified_runs.append((idx, save_dir))
            self.multi_run_manager.ready_to_update[idx] = False
        return notified_runs

    def _wait_for_nixl_ready(self, notified_runs: list[tuple[int, Path]]) -> None:
        for idx, save_dir in notified_runs:
            sync_wait_for_path(save_dir / NIXL_READY_MARKER, interval=0.1, log_interval=10)
