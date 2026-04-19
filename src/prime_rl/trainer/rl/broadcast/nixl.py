"""Trainer-side NIXL (UCX/RDMA) weight sender.

Replaces the per-step ``.full_tensor()``/NCCL-broadcast path for GLM MoE DSA
training. Each trainer rank pushes only its owned FSDP/EP shards directly
into pre-registered parameter memory on the inference side.

Sketch-level v1 assumptions (see design doc for full rollout):
  * GLM MoE DSA only (uses ``convert_tt_layer_to_vllm_kernel``).
  * FP8 kernel-format transfer (``quantize_in_weight_transfer`` is implicit).
  * Divisible trainer/inference world sizes (enforced by ``konig_schedule``).

Expert ownership is computed from ``parallel_dims``, not from the global rank:
in prime-rl the ``ep`` degree is **borrowed** from ``dp_shard * cp`` (see
``parallel_dims._build_mesh_with_ep``), so:

  * the EP group is the flattened ``dp_shard_in_ep × cp`` submesh, and
  * ``dp_shard_mod_ep × dp_replicate × pp`` is the replication factor — every
    expert's weights live on multiple trainer ranks.

Only one representative per replica group should push to inference; we pick
the rank whose (dp_shard_mod_ep, dp_replicate, pp) coordinate is all-zero.
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
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl import filter_state_dict_by_layers
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nixl_transfer import NixlAgentWrapper, konig_schedule, make_agent_name
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix

NIXL_READY_MARKER = "NIXL_READY"

# Conversion is delegated to the GLM-specific helper for v1.
try:
    from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel
except ImportError:  # pragma: no cover
    convert_tt_layer_to_vllm_kernel = None  # type: ignore


def _to_local(tensor: Tensor) -> Tensor:
    """Return the rank-local shard of a DTensor, or the tensor itself if dense."""
    if isinstance(tensor, DTensor):
        return cast(DTensor, tensor).to_local()
    return tensor


def _materialize_local(state_dict: dict[str, Tensor], dtype: torch.dtype) -> dict[str, Tensor]:
    """Swap each DTensor in ``state_dict`` for its rank-local shard, cast to ``dtype``.

    Critically — unlike ``_resolve_dtensors`` in the NCCL path — this function
    does **not** call ``full_tensor()``; no cross-rank gather is incurred.
    """
    return {key: _to_local(value).to(dtype) for key, value in state_dict.items()}


class _StableLayerSlots:
    """Per-layer stable buffers sized to the trainer-local shard.

    For GLM MoE DSA each layer's MoE experts contribute four tensors that we
    register once with NIXL and reuse every step:

        * ``w13_weight``:       (num_local_experts, 2*moe_dim, dim)   fp8_e4m3fn
        * ``w2_weight``:        (num_local_experts, dim, moe_dim)     fp8_e4m3fn
        * ``w13_weight_scale_inv``: (num_local_experts, *scale)        float32
        * ``w2_weight_scale_inv``:  (num_local_experts, *scale)        float32

    Non-expert tensors are currently emitted fresh per step (TODO: extend the
    stable-slot allocator to cover them too once the non-expert layout is
    locked down).
    """

    def __init__(self) -> None:
        self.slots: dict[int, dict[str, Tensor]] = {}
        self.chunked_descs: dict[int, dict[str, Any]] = {}  # layer → name → nixl xfer descs
        self.local_preps: dict[int, dict[str, Any]] = {}

    def layer_out_buffers(self, layer_idx: int) -> dict[str, Tensor]:
        return self.slots.get(layer_idx, {})


def _allocate_expert_slots(
    layer_idx: int,
    num_local_experts: int,
    moe_dim: int,
    dim: int,
    device: torch.device,
) -> dict[str, Tensor]:
    w13_shape = (num_local_experts, 2 * moe_dim, dim)
    w2_shape = (num_local_experts, dim, moe_dim)
    s_w13 = (ceil_div(2 * moe_dim, BLOCK_SIZE), ceil_div(dim, BLOCK_SIZE))
    s_w2 = (ceil_div(dim, BLOCK_SIZE), ceil_div(moe_dim, BLOCK_SIZE))
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.mlp.experts.w13_weight": torch.zeros(w13_shape, dtype=torch.float8_e4m3fn, device=device),
        f"{prefix}.mlp.experts.w2_weight": torch.zeros(w2_shape, dtype=torch.float8_e4m3fn, device=device),
        f"{prefix}.mlp.experts.w13_weight_scale_inv": torch.zeros(
            (num_local_experts, *s_w13), dtype=torch.float32, device=device
        ),
        f"{prefix}.mlp.experts.w2_weight_scale_inv": torch.zeros(
            (num_local_experts, *s_w2), dtype=torch.float32, device=device
        ),
    }


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
        if convert_tt_layer_to_vllm_kernel is None:
            raise RuntimeError("GLM MoE DSA converter not available — required for NIXL transfer")

        self.logger = get_logger()
        self.config = config
        self.world = get_world()
        self.parallel_dims = parallel_dims
        self._multi_run_manager = None  # lazy, to keep this class usable outside the trainer loop
        self.device = torch.device(device)
        self.dtype = dtype

        self.trainer_ws = self.world.world_size
        self.inference_ws = config.inference_world_size

        self._agent = NixlAgentWrapper(
            name=make_agent_name("trainer", self.world.rank),
            local_rank=self.world.local_rank,
            backends=config.backends,
        )

        # Discover moe_dim / dim / num_experts / num_layers from the model config.
        mc = model.config
        self.num_layers: int = mc.num_hidden_layers
        self.num_experts: int = getattr(mc, "n_routed_experts", getattr(mc, "num_experts", 0))
        self.moe_dim: int = getattr(mc, "moe_intermediate_size", getattr(mc, "intermediate_size"))
        self.hidden_dim: int = mc.hidden_size
        self.first_k_dense: int = getattr(mc, "first_k_dense_replace", 0)

        # EP mesh — in prime-rl, ep is borrowed from dp_shard_in_ep * cp; the actual EP
        # world size and the rank's EP coordinate come from the flattened "ep" submesh.
        if parallel_dims.ep_enabled:
            ep_mesh = parallel_dims.get_mesh("ep")
            self.ep_size = ep_mesh.size()
            self.ep_rank = ep_mesh.get_local_rank()
        else:
            self.ep_size = 1
            self.ep_rank = 0

        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by EP size ({self.ep_size})"
            )
        self.num_local_experts = self.num_experts // self.ep_size
        self.owned_global_experts = list(
            range(self.ep_rank * self.num_local_experts, (self.ep_rank + 1) * self.num_local_experts)
        )

        # Every expert shard is replicated across (dp_shard_mod_ep × dp_replicate × pp).
        # Elect a single "lead" rank per replica group to avoid duplicate writes.
        # The lead is the rank whose non-EP coordinate is 0 along every replicated dim.
        self._is_expert_lead = self._compute_is_expert_lead()

        # Allocate stable expert slots per MoE layer and register them with NIXL.
        self._slots = _StableLayerSlots()
        layer_descs: dict[str, bytes] = {}
        for layer_idx in range(self.first_k_dense, self.num_layers):
            slots = _allocate_expert_slots(
                layer_idx,
                self.num_local_experts,
                self.moe_dim,
                self.hidden_dim,
                self.device,
            )
            self._slots.slots[layer_idx] = slots
            self._slots.chunked_descs[layer_idx] = {}
            self._slots.local_preps[layer_idx] = {}
            for name, tensor in slots.items():
                # Register the whole slot so trainer + inference can address it.
                self._agent.register_tensor(tensor)
                # Build per-expert chunks (one chunk per local expert).
                chunk_descs = self._agent.chunked_descs(tensor, self.num_local_experts)
                self._slots.chunked_descs[layer_idx][name] = chunk_descs
                self._slots.local_preps[layer_idx][name] = self._agent.prep_local(chunk_descs)
                layer_descs[name] = self._agent.serialize_descs(chunk_descs)

        # Rendezvous
        full_ws = self.trainer_ws + self.inference_ws
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
            "descriptors": layer_descs,  # per-tensor chunked descs (one chunk = one owned expert)
            "owned_global_experts": self.owned_global_experts,
        }
        all_info: list[dict[str, Any]] = self._spg.all_gather_obj(my_info)

        inference_infos = all_info[self.trainer_ws :]
        # Add every inference peer, deserialize their per-tensor descs, prep remote dlists.
        self._expert_routes: dict[tuple[int, str, int], tuple[Any, int]] = {}
        for peer in inference_infos:
            self._agent.add_remote(peer["agent_metadata"])

        # Build routing table: for each (layer, tensor_name, local_expert_idx in this trainer),
        # find (remote_prep, remote_slot_idx) on the inference rank that owns that global expert.
        for layer_idx in range(self.first_k_dense, self.num_layers):
            for tensor_name in self._slots.slots[layer_idx].keys():
                moe_module_prefix = f"model.layers.{layer_idx}.mlp.experts"
                for local_idx, global_idx in enumerate(self.owned_global_experts):
                    for peer in inference_infos:
                        expert_map: dict[str, list[int]] = peer["expert_map"]
                        global_list = expert_map.get(moe_module_prefix)
                        if global_list is None:
                            continue
                        if global_idx not in global_list:
                            continue
                        remote_slot_idx = global_list.index(global_idx)
                        remote_descs = self._agent.deserialize_descs(peer["descriptors"][tensor_name])
                        remote_prep = self._agent.prep_remote(peer["agent_name"], remote_descs)
                        self._expert_routes[(layer_idx, tensor_name, local_idx)] = (remote_prep, remote_slot_idx)
                        break

        # Eager connection wire-up (playground-style) so the first broadcast isn't racing UCX setup.
        for peer in inference_infos:
            self._agent.make_connection(peer["agent_name"])

        # König schedule for non-expert tensors (currently unused in v1 write loop — TODO).
        self._konig = konig_schedule(self.trainer_ws, self.inference_ws)

        # Bytes this rank will send on every push (sum of all stable slot sizes this rank writes).
        # Expert-lead ranks send the full per-layer expert payload; non-leads send nothing.
        self._bytes_per_push = 0
        if self._is_expert_lead:
            for layer_idx in range(self.first_k_dense, self.num_layers):
                for tensor in self._slots.slots[layer_idx].values():
                    self._bytes_per_push += tensor.numel() * tensor.element_size()

        self.logger.info(
            f"NIXL transfer initialized: rank={self.world.rank} "
            f"owned_experts={self.owned_global_experts} routes={len(self._expert_routes)} "
            f"bytes_per_push={self._bytes_per_push / 1e6:.2f} MB"
        )

    @torch.no_grad()
    def push_once(self, model: nn.Module) -> None:
        """Core NIXL weight push — no orchestrator notification or multi-run bookkeeping.

        Useful for integration tests and anywhere a caller drives the handshake
        manually. :meth:`broadcast_weights` wraps this with the full pause/
        ready-marker dance.

        Logs a per-rank throughput breakdown (convert / post / wait / barrier,
        plus GB/s over the payload this rank sent) at INFO level every call.
        """
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)

        t_start = time.perf_counter()

        # Pass 1 — convert every layer into its stable FP8 slot. These kernels
        # run on the default CUDA stream; we *must* synchronize before posting
        # any NIXL writes, otherwise UCX reads HBM addresses whose contents the
        # GPU hasn't finished writing yet (silent scale/weight corruption).
        moe_layers: list[int] = []
        for layer_idx, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            if layer_idx < self.first_k_dense:
                continue
            local_sd = _materialize_local(layer_state_dict, self.dtype)
            convert_tt_layer_to_vllm_kernel(
                local_sd,
                layer_idx,
                out_buffers=self._slots.layer_out_buffers(layer_idx),
            )
            moe_layers.append(layer_idx)
        torch.cuda.synchronize(self.device)
        t_converted = time.perf_counter()

        # Pass 2 — post NIXL writes.
        handles: list[Any] = []
        if self._is_expert_lead:
            for layer_idx in moe_layers:
                local_preps = self._slots.local_preps[layer_idx]
                for tensor_name, local_prep in local_preps.items():
                    for local_idx in range(self.num_local_experts):
                        key = (layer_idx, tensor_name, local_idx)
                        route = self._expert_routes.get(key)
                        if route is None:
                            self.logger.warning(f"No route for {key}, skipping")
                            continue
                        remote_prep, remote_slot_idx = route
                        handles.append(
                            self._agent.post_write(local_prep, local_idx, remote_prep, remote_slot_idx)
                        )
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
        dt_wire = t_waited - t_start  # convert + post + wait, excludes final barrier

        gbps_wire = (self._bytes_per_push / dt_wire / 1e9) if dt_wire > 0 else 0.0
        gbps_post_wait = (
            (self._bytes_per_push / (dt_post + dt_wait) / 1e9) if (dt_post + dt_wait) > 0 else 0.0
        )
        gbps_total = (self._bytes_per_push / dt_total / 1e9) if dt_total > 0 else 0.0

        self.logger.info(
            f"[nixl rank={self.world.rank}] push "
            f"bytes={self._bytes_per_push / 1e6:.2f}MB "
            f"handles={len(handles)} "
            f"convert={dt_convert * 1e3:.2f}ms "
            f"post={dt_post * 1e3:.2f}ms "
            f"wait={dt_wait * 1e3:.2f}ms "
            f"barrier={dt_barrier * 1e3:.2f}ms "
            f"total={dt_total * 1e3:.2f}ms "
            f"wire_bw={gbps_wire:.2f}GB/s net_bw={gbps_post_wait:.2f}GB/s total_bw={gbps_total:.2f}GB/s"
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        start = time.perf_counter()
        self.logger.debug("Starting NIXL weight broadcast")

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

    def _compute_is_expert_lead(self) -> bool:
        """True when this rank is the designated expert-writer for its EP shard.

        Expert weights are replicated across ``dp_shard_mod_ep × dp_replicate × pp``;
        we pick the replica whose coordinate along each of those dims is 0.
        """
        pd = self.parallel_dims
        for dim_name in ("dp_shard_mod_ep", "dp_replicate", "pp"):
            if getattr(pd, dim_name, 1) > 1:
                try:
                    submesh = pd.get_mesh(dim_name)
                except (KeyError, RuntimeError):
                    continue
                if submesh.get_local_rank() != 0:
                    return False
        return True

    def _notify_orchestrator(self) -> list[tuple[int, Path]]:
        notified_runs: list[tuple[int, Path]] = []
        for idx in self.multi_run_manager.used_idxs:
            if not self.multi_run_manager.ready_to_update[idx]:
                continue
            try:
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step,
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / "STABLE").touch()
                notified_runs.append((idx, save_dir))
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} is deleted, skipping")
            except Exception as e:
                self.logger.error(f"Error notifying orchestrator for run {idx}: {e}")
            finally:
                self.multi_run_manager.ready_to_update[idx] = False
        return notified_runs

    def _wait_for_nixl_ready(self, notified_runs: list[tuple[int, Path]]) -> None:
        for idx, save_dir in notified_runs:
            marker = save_dir / NIXL_READY_MARKER
            self.logger.debug(f"Waiting for NIXL_READY marker at {marker}")
            sync_wait_for_path(marker, interval=0.1, log_interval=10)
            self.logger.debug(f"Inference workers ready for NIXL writes (run {idx})")
