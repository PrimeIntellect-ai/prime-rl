"""Trainer-side NIXL (UCX/RDMA) weight sender.

Replaces the per-step ``.full_tensor()``/NCCL-broadcast path for GLM MoE DSA
training: each trainer rank pushes its FSDP/EP shards directly into pre-
registered parameter memory on the inference side.

Every destination buffer the converter writes to is allocated up front and
registered with NIXL. Expert tensors get per-expert chunked descriptors (so
``local_expert[i]`` can land in the inference rank that owns the matching
global expert); non-expert tensors are registered as single-chunk handles.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import (
    _BASE,
    _DENSE,
    _SPARSE,
    convert_tt_layer_to_vllm_kernel,
)
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nixl_transfer import NixlAgentWrapper, make_agent_name
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix

NIXL_READY_MARKER = "NIXL_READY"


def _to_local(tensor: Tensor) -> Tensor:
    return cast(DTensor, tensor).to_local() if isinstance(tensor, DTensor) else tensor


def _materialize_local(state_dict: dict[str, Tensor], dtype: torch.dtype) -> dict[str, Tensor]:
    return {key: _to_local(value).to(dtype) for key, value in state_dict.items()}


def _allocate_layer_slots(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Tensor]:
    """Pre-allocate every destination buffer ``convert_tt_layer_to_vllm_kernel`` writes.

    Branches once on dense vs sparse (same rule the converter uses). Expert
    tensors come in sharded along the ep dim, so ``to_local()`` gives the
    rank-local ``(num_local_experts, …)`` shape. Quantized destinations also
    get a float32 ``_scale_inv`` companion sized for 128×128 blocks.
    """
    prefix = f"model.layers.{layer_idx}"
    is_sparse = f"{prefix}.mlp.router.gate.weight" in state_dict
    specs = _BASE + (_SPARSE if is_sparse else _DENSE)

    slots: dict[str, Tensor] = {}
    for spec in specs:
        srcs = [_to_local(state_dict[f"{prefix}.{s}"]) for s in spec.sources]
        dst_shape = list(srcs[0].shape)
        dst_shape[spec.cat_dim] *= len(srcs)
        dst_dtype = torch.float8_e4m3fn if spec.quantize else dtype
        slots[f"{prefix}.{spec.dst}"] = torch.empty(dst_shape, dtype=dst_dtype, device=device)
        if spec.quantize:
            scale_shape = tuple(
                ceil_div(d, BLOCK_SIZE) if i >= len(dst_shape) - 2 else d for i, d in enumerate(dst_shape)
            )
            slots[spec.scale_name(prefix)] = torch.empty(scale_shape, dtype=torch.float32, device=device)
    return slots


def _is_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name


class NIXLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via NIXL (zero-copy RDMA)."""

    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        model: nn.Module,
        device: torch.device,
        parallel_dims: ParallelDims,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(output_dir)
        self.logger = get_logger()
        self.config = config
        self.world = get_world()
        self.device = torch.device(device)
        self.dtype = dtype
        self._multi_run_manager = None

        self.num_layers = model.config.num_hidden_layers
        if parallel_dims.ep_enabled:
            ep_mesh = parallel_dims.get_mesh("ep")
            ep_size, ep_rank = ep_mesh.size(), ep_mesh.get_local_rank()
        else:
            ep_size, ep_rank = 1, 0
        self.num_local_experts = model.config.n_routed_experts // ep_size
        owned_global_experts = list(
            range(ep_rank * self.num_local_experts, (ep_rank + 1) * self.num_local_experts)
        )

        self._agent = NixlAgentWrapper(
            name=make_agent_name("trainer", self.world.rank),
            local_rank=self.world.local_rank,
            backends=config.backends,
        )

        # Allocate every destination buffer the converter writes into. The
        # converter mutates these in place; ``push_once`` reads them back.
        model_sd = model.state_dict()
        self._slots: dict[int, dict[str, Tensor]] = {
            i: _allocate_layer_slots(model_sd, i, self.dtype, self.device) for i in range(self.num_layers)
        }

        # Register every slot with NIXL. Expert slots get chunked per-expert so
        # ``local_expert[i]`` can land in whichever inference rank owns it;
        # non-expert slots are single-chunk.
        descriptors: dict[str, bytes] = {}
        local_preps: dict[str, Any] = {}
        for layer_idx in range(self.num_layers):
            for name, tensor in self._slots[layer_idx].items():
                self._agent.register_tensor(tensor)
                chunks = self.num_local_experts if _is_expert_tensor(name) else 1
                descs = self._agent.chunked_descs(tensor, chunks)
                descriptors[name] = self._agent.serialize_descs(descs)
                local_preps[name] = self._agent.prep_local(descs)

        # Rendezvous.
        full_ws = self.world.world_size + config.inference_world_size
        self._spg = StatelessProcessGroup.create(
            host=config.host,
            port=config.port,
            rank=self.world.rank,
            world_size=full_ws,
            store_timeout=config.timeout,
        )
        my_info = {
            "role": "trainer",
            "global_rank": self.world.rank,
            "agent_name": self._agent.name,
            "agent_metadata": self._agent.get_metadata(),
            "descriptors": descriptors,
            "owned_global_experts": owned_global_experts,
        }
        all_info: list[dict[str, Any]] = self._spg.all_gather_obj(my_info)
        inference_infos = all_info[self.world.world_size :]
        for peer in inference_infos:
            self._agent.add_remote(peer["agent_metadata"])
            self._agent.make_connection(peer["agent_name"])

        # Build per-expert write routes. Each trainer rank's EP shard is
        # unique, so no "lead" election is needed. ``next(...)`` without a
        # default is the fail-loud assertion that every owned expert has
        # exactly one home on the inference side.
        self._writes: list[tuple[Any, int, Any, int]] = []
        for layer_idx in range(self.num_layers):
            for name in self._slots[layer_idx]:
                if not _is_expert_tensor(name):
                    continue
                moe_prefix = f"model.layers.{layer_idx}.mlp.experts"
                local_prep = local_preps[name]
                for local_idx, global_idx in enumerate(owned_global_experts):
                    peer = next(p for p in inference_infos if global_idx in p["expert_map"][moe_prefix])
                    remote_idx = peer["expert_map"][moe_prefix].index(global_idx)
                    remote_descs = self._agent.deserialize_descs(peer["descriptors"][name])
                    remote_prep = self._agent.prep_remote(peer["agent_name"], remote_descs)
                    self._writes.append((local_prep, local_idx, remote_prep, remote_idx))

        self._bytes_per_push = sum(
            t.numel() * t.element_size()
            for slots in self._slots.values()
            for name, t in slots.items()
            if _is_expert_tensor(name)
        )

        self.logger.info(
            f"NIXL transfer initialized: rank={self.world.rank} "
            f"owned_experts={owned_global_experts} writes={len(self._writes)} "
            f"bytes_per_push={self._bytes_per_push / 1e6:.2f} MB"
        )

    @torch.no_grad()
    def push_once(self, model: nn.Module) -> None:
        """Convert every layer into its stable FP8 slot and post all writes."""
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)

        t_start = time.perf_counter()

        # Pass 1 — convert. Must synchronize before posting NIXL writes, or
        # UCX reads HBM whose contents the GPU hasn't finished writing.
        for layer_idx in range(self.num_layers):
            layer_sd = {k: v for k, v in state_dict.items() if k.startswith(f"{layer_prefix}{layer_idx}.")}
            convert_tt_layer_to_vllm_kernel(
                _materialize_local(layer_sd, self.dtype),
                layer_idx,
                out_buffers=self._slots[layer_idx],
            )
        torch.cuda.synchronize(self.device)
        t_converted = time.perf_counter()

        # Pass 2 — post writes.
        handles = [self._agent.post_write(lp, li, rp, ri) for lp, li, rp, ri in self._writes]
        t_posted = time.perf_counter()

        for h in handles:
            self._agent.wait(h)
        t_waited = time.perf_counter()

        self._spg.barrier()
        t_done = time.perf_counter()

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
