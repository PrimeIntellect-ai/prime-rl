"""vLLM worker extension for composed, sharded NIXL weight pulls."""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from math import prod
from threading import Event
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import update_mla_absorbed_weights
from prime_rl.weight_transfer.chains import (
    OpChain,
    apply_chain,
    region_elem_runs,
    resolve_chain_region,
    split_transport_chain,
    tensor_runs,
)
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc, cuda_buffer_capacity
from prime_rl.weight_transfer.graph import make_hf_lazy_weights
from prime_rl.weight_transfer.lazy import BakeRecorder, RecordedCopy
from prime_rl.weight_transfer.mx import MxRendezvous
from prime_rl.weight_transfer.nixl import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.weight_transfer.sharding import route_region, zip_src_dst
from prime_rl.weight_transfer.wire import TrainerTable

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")
_BUFFER_POLL_INTERVAL = 0.01


@dataclass
class TensorCopySpec:
    transport_ops: OpChain
    replay_ops: OpChain
    staging_shape: tuple[int, ...]


@dataclass
class TensorCopyPlan:
    recorded_copy: RecordedCopy
    staging_tensor: torch.Tensor
    replay_ops: OpChain


@dataclass
class LayerWeightCopies:
    layer: nn.Module
    copies: list[RecordedCopy]


@dataclass
class LayerWeightTransferPlan:
    reload_layer: nn.Module | None
    copies: list[TensorCopyPlan]
    persistent_copies: list[TensorCopyPlan]

    @property
    def destination_names(self) -> set[str]:
        return {plan.recorded_copy.param_name for plan in self.copies}


@dataclass
class WeightTransferGroup:
    name: str
    layers: list[LayerWeightTransferPlan]
    pulls: list[tuple[Any, Any, list[int]]]


@dataclass
class WeightTransferPlan:
    receive_arenas: dict[torch.dtype, torch.Tensor]
    receive_buffer_count: int
    groups: list[WeightTransferGroup]
    total_bytes: int


@dataclass
class PulledWeightTransferGroup:
    group: WeightTransferGroup
    source_wait_seconds: float
    transfer_seconds: float
    acknowledgement_seconds: float


class NIXLWeightUpdateWorker(Worker):
    @property
    def raw_model(self) -> nn.Module:
        return cast(nn.Module, self.model_runner.get_model())

    def liveness_probe(self) -> None:
        return None

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
        session_id: str = "default",
    ) -> None:
        del inference_world_size, quantize_in_weight_transfer
        global_rank = rank_offset + int(self.local_rank)
        server_url = f"{host}:{port}"
        set_ucx_env_defaults()
        self.nixl_agent = NixlAgent(make_agent_name("inference", global_rank))
        self.mx_rendezvous = MxRendezvous(
            client=MxClient(server_url=server_url),
            role="inference",
            rank=global_rank,
            peer_world_size=1,
            session_id=session_id,
            worker_id=f"inference-{global_rank}",
        )
        self.weight_transfer_timeout = timeout
        self.weight_transfer_plan: WeightTransferPlan | None = None
        logger.info(
            "NIXL worker configured: global_rank=%d, ModelExpress=%s, session=%s",
            global_rank,
            server_url,
            session_id,
        )

    @torch.no_grad()
    def _lazy_init(self) -> WeightTransferPlan:
        if self.weight_transfer_plan is not None:
            return self.weight_transfer_plan

        started = time.perf_counter()
        allocated_bytes = torch.cuda.memory_allocated(self.device)
        peak_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        trainer_ref = self.mx_rendezvous.wait_for_peers(timeout=self.weight_transfer_timeout)[0]
        table = TrainerTable.decode(self.mx_rendezvous.fetch(trainer_ref).nixl_metadata)
        layers, persistent = self._bake(table)
        plan = self._build_pull_plan(table, layers, persistent, allocated_bytes, peak_allocated_bytes)
        self.buffer_rendezvous = []
        for buffer_index in range(table.source_ring_size):
            rendezvous = MxRendezvous(
                client=self.mx_rendezvous.client,
                role="inference",
                rank=self.mx_rendezvous.rank,
                peer_world_size=1,
                session_id=f"{self.mx_rendezvous.session_id}:layers:{buffer_index}",
                worker_id=f"inference-buffer-{self.mx_rendezvous.rank}-{buffer_index}",
            )
            rendezvous.publish()
            rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            self.buffer_rendezvous.append(rendezvous)
        # Join the current generation directly. Publishing a transient READY
        # before the first pull would let the trainer mistake initialization
        # for a completed acknowledgement.
        self.mx_rendezvous.publish(nixl_metadata=self.nixl_agent.get_metadata())
        self.mx_rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.weight_transfer_plan = plan
        logger.info(
            "NIXL plan baked in %.2fs: rank=%d, groups=%d, source_buffers=%d, copies=%d, bytes=%d, pull_lists=%d",
            time.perf_counter() - started,
            self.mx_rendezvous.rank,
            len(plan.groups),
            table.source_ring_size,
            sum(len(layer.copies) + len(layer.persistent_copies) for group in plan.groups for layer in group.layers),
            plan.total_bytes,
            sum(len(group.pulls) for group in plan.groups),
        )
        return plan

    def _bake(
        self,
        table: TrainerTable,
    ) -> tuple[list[LayerWeightCopies], list[RecordedCopy]]:
        from vllm.model_executor.model_loader.reload.layerwise import (
            _get_original_loader,
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        model = self.raw_model
        recorder = BakeRecorder()
        persistent: list[RecordedCopy] = []
        original_loaders: list[tuple[torch.Tensor, Any]] = []
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            try:
                for module in model.modules():
                    for name, tensor in get_layer_tensors(module).items():
                        if not tensor.is_meta:
                            recorder.register_live_destination(module, name, tensor)
                        loader = _get_original_loader(tensor)
                        original_loaders.append((tensor, loader))
                        tensor.weight_loader = self._stamp(recorder, module, name, loader)

                model.load_weights(
                    make_hf_lazy_weights(
                        table,
                        device=self.device,
                        recorder=recorder,
                        hf_config=self.model_runner.model_config.hf_text_config,
                    )
                )

                by_layer: dict[int, list[RecordedCopy]] = defaultdict(list)
                layers: dict[int, nn.Module] = {}
                for copy in recorder.copies:
                    if copy.persistent or copy.param_name in SKIP_TENSORS:
                        copy.persistent = True
                        persistent.append(copy)
                    else:
                        by_layer[id(copy.layer)].append(copy)
                        layers[id(copy.layer)] = copy.layer
            finally:
                try:
                    for tensor, loader in reversed(original_loaders):
                        tensor.weight_loader = loader
                finally:
                    self._restore_layerwise_state(model)

        return [
            LayerWeightCopies(layer=layers[layer_id], copies=copies) for layer_id, copies in by_layer.items()
        ], persistent

    @staticmethod
    def _stamp(recorder: BakeRecorder, layer: nn.Module, name: str, loader: Any):
        @wraps(loader)
        def stamped(*args, **kwargs):
            recorder.current = (layer, name)
            try:
                return loader(*args, **kwargs)
            finally:
                recorder.current = None

        return stamped

    @staticmethod
    def _restore_layerwise_state(model: nn.Module) -> None:
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO, _place_kernel_tensors

        for layer in model.modules():
            info = LAYERWISE_INFO.get(layer)
            if info is not None and info.can_load():
                if info.kernel_tensors is not None:
                    _place_kernel_tensors(layer, info)
                info.reset()
        if hasattr(model, "_original_do_torchao_reload"):
            model._do_torchao_reload = model._original_do_torchao_reload

    def _build_pull_plan(
        self,
        table: TrainerTable,
        layers: list[LayerWeightCopies],
        persistent: list[RecordedCopy],
        allocated_bytes: int,
        peak_allocated_bytes: int,
    ) -> WeightTransferPlan:
        tensors = {tensor.name: tensor for group in table.groups for tensor in group.tensors}
        tensor_groups = {
            tensor.name: group_index
            for group_index, group in enumerate(table.groups)
            for tensor in group.tensors
        }
        copies = [copy for layer in layers for copy in layer.copies] + persistent
        specifications: dict[int, TensorCopySpec] = {}
        copies_by_group: dict[int, list[RecordedCopy]] = defaultdict(list)
        group_elements: dict[torch.dtype, list[int]] = defaultdict(lambda: [0] * len(table.groups))
        for copy in copies:
            source = tensors[copy.src_name]
            source_dtype = getattr(torch, source.wire_dtype)
            transport_ops, replay_ops, staging_shape = split_transport_chain(
                tuple(source.shape), source_dtype, copy.ops
            )
            specifications[id(copy)] = TensorCopySpec(
                transport_ops=transport_ops,
                replay_ops=replay_ops,
                staging_shape=staging_shape,
            )
            source_group = tensor_groups[source.name]
            copies_by_group[source_group].append(copy)
            group_elements[source_dtype][source_group] += prod(staging_shape)

        reload_layer_ids = {id(layer.layer) for layer in layers}
        reload_layer_groups: dict[int, int] = {}
        for copy in copies:
            layer_id = id(copy.layer)
            if layer_id not in reload_layer_ids:
                continue
            source_group = tensor_groups[copy.src_name]
            previous_group = reload_layer_groups.setdefault(layer_id, source_group)
            if previous_group != source_group:
                raise RuntimeError(
                    f"vLLM reload layer {type(copy.layer).__name__} reads trainer groups "
                    f"{table.groups[previous_group].name!r} and {table.groups[source_group].name!r}"
                )

        largest_group_elements = {dtype: max(elements, default=0) for dtype, elements in group_elements.items()}
        largest_group_bytes = max(
            1,
            sum(elements * dtype.itemsize for dtype, elements in largest_group_elements.items()),
        )
        peak_growth_bytes = max(0, peak_allocated_bytes - allocated_bytes)
        has_observed_peak_growth = peak_growth_bytes > 0
        free_before_reclaim, _ = torch.cuda.mem_get_info(self.device)
        max_receive_buffers = min(2, table.source_ring_size) if has_observed_peak_growth else 1
        if has_observed_peak_growth or free_before_reclaim < largest_group_bytes:
            torch.cuda.empty_cache()
        receive_buffer_count, free_bytes, device_total_bytes, headroom_bytes = cuda_buffer_capacity(
            largest_group_bytes,
            max_receive_buffers,
            self.device,
            extra_headroom_bytes=largest_group_bytes + peak_growth_bytes,
        )
        with classic_cuda_alloc():
            receive_arenas = {
                dtype: torch.empty(
                    receive_buffer_count * elements,
                    dtype=dtype,
                    device=self.device,
                )
                for dtype, elements in largest_group_elements.items()
                if elements
            }
        for arena in receive_arenas.values():
            self.nixl_agent.register_tensor(arena)
        receive_arena_bytes = sum(arena.nbytes for arena in receive_arenas.values())
        post_free_bytes, _ = torch.cuda.mem_get_info(self.device)

        agent_devices = {agent_index: agent.device_id for agent_index, agent in enumerate(table.agents)}
        total_bytes = 0
        peer_names: dict[int, str] = {}
        transfer_groups: list[WeightTransferGroup] = []

        for group_index, group in enumerate(table.groups):
            copy_plans_by_layer: dict[int, list[TensorCopyPlan]] = defaultdict(list)
            persistent_plans_by_layer: dict[int, list[TensorCopyPlan]] = defaultdict(list)
            local_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            remote_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            cursors = {
                dtype: (group_index % receive_buffer_count) * elements
                for dtype, elements in largest_group_elements.items()
            }

            for copy in copies_by_group[group_index]:
                specification = specifications[id(copy)]
                source = tensors[copy.src_name]
                source_dtype = getattr(torch, source.wire_dtype)
                numel = prod(specification.staging_shape)
                cursor = cursors[source_dtype]
                staging_tensor = receive_arenas[source_dtype].narrow(0, cursor, numel).view(specification.staging_shape)
                cursors[source_dtype] += numel
                copy_plan = TensorCopyPlan(
                    recorded_copy=copy,
                    staging_tensor=staging_tensor,
                    replay_ops=specification.replay_ops,
                )
                plans = persistent_plans_by_layer if copy.persistent else copy_plans_by_layer
                plans[id(copy.layer)].append(copy_plan)

                offset, shape, stride = resolve_chain_region(
                    tuple(source.shape), source_dtype, specification.transport_ops
                )
                source_pieces = route_region(
                    region_elem_runs(offset, shape, stride),
                    source.shards,
                    prod(source.shape),
                    source_dtype.itemsize,
                )
                for agent, source_addr, destination_addr, nbytes in zip_src_dst(
                    source_pieces, tensor_runs(staging_tensor)
                ):
                    local_descs[agent].append((destination_addr, nbytes, self.device.index))
                    remote_descs[agent].append((source_addr, nbytes, agent_devices[agent]))
                    total_bytes += nbytes

            layer_plans: list[LayerWeightTransferPlan] = []
            for layer in layers:
                layer_id = id(layer.layer)
                regular = copy_plans_by_layer.pop(layer_id, [])
                live = persistent_plans_by_layer.pop(layer_id, [])
                if regular:
                    layer_plans.append(
                        LayerWeightTransferPlan(
                            reload_layer=layer.layer,
                            copies=regular,
                            persistent_copies=live,
                        )
                    )
                elif live:
                    layer_plans.append(
                        LayerWeightTransferPlan(
                            reload_layer=None,
                            copies=[],
                            persistent_copies=live,
                        )
                    )
            remaining_persistent = [plan for plans in persistent_plans_by_layer.values() for plan in plans]
            if remaining_persistent:
                layer_plans.append(
                    LayerWeightTransferPlan(
                        reload_layer=None,
                        copies=[],
                        persistent_copies=remaining_persistent,
                    )
                )

            pulls: list[tuple[Any, Any, list[int]]] = []
            for agent_index, remote in sorted(remote_descs.items()):
                peer_name = peer_names.get(agent_index)
                if peer_name is None:
                    peer_name = self.nixl_agent.add_remote_agent(table.agents[agent_index].metadata)
                    self.nixl_agent.make_connection(peer_name)
                    peer_names[agent_index] = peer_name
                local_prepared = self.nixl_agent.prep_local(local_descs[agent_index])
                remote_prepared = self.nixl_agent.prep_remote(peer_name, remote)
                pulls.append((local_prepared, remote_prepared, list(range(len(remote)))))
            transfer_groups.append(
                WeightTransferGroup(
                    name=group.name,
                    layers=layer_plans,
                    pulls=pulls,
                )
            )

        logger.info(
            "NIXL receive ring selected %d buffers from one-time first-transfer sizing: "
            "rank=%d, peak_mode=%s, trainer_buffers=%d, local_cap=%d, "
            "%.2f GiB CUDA free before and %.2f GiB after cache reclaim; "
            "active=%.2f GiB, peak=%.2f GiB, peak_growth=%.2f GiB "
            "(%.2f GiB total, %.2f GiB target headroom): %.2f GiB per buffer, "
            "%.2f GiB arena, %.2f GiB post-allocation free for %d groups",
            receive_buffer_count,
            self.mx_rendezvous.rank,
            "measured" if has_observed_peak_growth else "unmeasured",
            table.source_ring_size,
            max_receive_buffers,
            free_before_reclaim / 1024**3,
            free_bytes / 1024**3,
            allocated_bytes / 1024**3,
            peak_allocated_bytes / 1024**3,
            peak_growth_bytes / 1024**3,
            device_total_bytes / 1024**3,
            headroom_bytes / 1024**3,
            largest_group_bytes / 1024**3,
            receive_arena_bytes / 1024**3,
            post_free_bytes / 1024**3,
            len(transfer_groups),
        )
        return WeightTransferPlan(
            receive_arenas=receive_arenas,
            receive_buffer_count=receive_buffer_count,
            groups=transfer_groups,
            total_bytes=total_bytes,
        )

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        del weight_dir
        plan = self._lazy_init()
        self.mx_rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.mx_rendezvous.wait_for(
            "trainer",
            count=1,
            status=p2p_pb2.SOURCE_STATUS_READY,
            timeout=self.weight_transfer_timeout,
        )

        started = time.perf_counter()
        self._process_and_commit(plan)
        update_mla_absorbed_weights(self.raw_model)
        torch.cuda.synchronize(self.device)
        self.mx_rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
        logger.info(
            "Applied %.2f GB NIXL policy update on rank %d in %.2fs",
            plan.total_bytes / 1e9,
            self.mx_rendezvous.rank,
            time.perf_counter() - started,
        )

    def _process_and_commit(self, plan: WeightTransferPlan) -> None:
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.meta import materialize_layer
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        model = self.raw_model
        cancelled = Event()

        def pull_group(group_index: int) -> PulledWeightTransferGroup:
            transfer_group = plan.groups[group_index]
            rendezvous = self.buffer_rendezvous[group_index % len(self.buffer_rendezvous)]
            source_wait_started = time.perf_counter()
            rendezvous.wait_for(
                "trainer",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.weight_transfer_timeout,
                poll_interval=_BUFFER_POLL_INTERVAL,
                cancelled=cancelled.is_set,
            )
            source_wait_seconds = time.perf_counter() - source_wait_started

            transfer_started = time.perf_counter()
            for local, remote, indices in transfer_group.pulls:
                handle = self.nixl_agent.post_read(local, indices, remote)
                self.nixl_agent.wait(
                    handle,
                    context=f"weight pull for {transfer_group.name}",
                    timeout=self.weight_transfer_timeout,
                    cancelled=cancelled.is_set,
                )
            transfer_seconds = time.perf_counter() - transfer_started

            return PulledWeightTransferGroup(
                group=transfer_group,
                source_wait_seconds=source_wait_seconds,
                transfer_seconds=transfer_seconds,
                acknowledgement_seconds=0.0,
            )

        def acknowledge_group(group_index: int) -> float:
            rendezvous = self.buffer_rendezvous[group_index % len(self.buffer_rendezvous)]
            acknowledgement_started = time.perf_counter()
            rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
            rendezvous.wait_for(
                "trainer",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.weight_transfer_timeout,
                poll_interval=_BUFFER_POLL_INTERVAL,
                cancelled=cancelled.is_set,
            )
            rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            return time.perf_counter() - acknowledgement_started

        def prefetch_group(group_index: int) -> PulledWeightTransferGroup:
            torch.cuda.set_device(self.device)
            pulled = pull_group(group_index)
            pulled.acknowledgement_seconds = acknowledge_group(group_index)
            return pulled

        def replay_group(transfer_group: WeightTransferGroup) -> None:
            for layer_plan in transfer_group.layers:
                layer = layer_plan.reload_layer
                if layer is None:
                    for copy_plan in layer_plan.persistent_copies:
                        self._copy_plan(copy_plan)
                    continue

                info = LAYERWISE_INFO[layer]
                materialize_layer(layer, info)
                # Match loader semantics for destinations with unwritten padding.
                destination_names = layer_plan.destination_names
                for name, tensor in get_layer_tensors(layer).items():
                    if name in destination_names and not tensor.is_meta:
                        tensor.zero_()
                for copy_plan in layer_plan.copies:
                    self._copy_plan(copy_plan)
                for copy_plan in layer_plan.persistent_copies:
                    self._copy_plan(copy_plan)

                if hasattr(layer, "_already_called_process_weights_after_loading"):
                    delattr(layer, "_already_called_process_weights_after_loading")
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    quant_method.process_weights_after_loading(layer)
                if info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(layer, info)
                info.reset()

        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            pipelined = plan.receive_buffer_count > 1
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nixl-prefetch") if pipelined else None
            source_wait_seconds = 0.0
            transfer_seconds = 0.0
            acknowledgement_seconds = 0.0
            replay_seconds = 0.0
            try:
                pull = executor.submit(prefetch_group, 0) if executor is not None else None
                for group_index in range(len(plan.groups)):
                    pulled = pull.result() if pull is not None else pull_group(group_index)
                    transfer_group = pulled.group
                    source_wait_seconds += pulled.source_wait_seconds
                    transfer_seconds += pulled.transfer_seconds
                    acknowledgement_seconds += pulled.acknowledgement_seconds

                    torch.cuda.synchronize(self.device)
                    if executor is not None and group_index + 1 < len(plan.groups):
                        pull = executor.submit(prefetch_group, group_index + 1)

                    replay_started = time.perf_counter()
                    replay_group(transfer_group)
                    torch.cuda.synchronize(self.device)
                    replay_seconds += time.perf_counter() - replay_started

                    if not pipelined:
                        acknowledgement_seconds += acknowledge_group(group_index)
            finally:
                cancelled.set()
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

            finalize_started = time.perf_counter()
            finalize_layerwise_reload(model, self.model_runner.model_config)
            logger.info(
                "NIXL update profile rank=%d: groups=%d, source_wait=%.2fs, "
                "sequential_rdma=%.2fs, source_ack=%.2fs, replay=%.2fs, finalize=%.2fs",
                self.mx_rendezvous.rank,
                len(plan.groups),
                source_wait_seconds,
                transfer_seconds,
                acknowledgement_seconds,
                replay_seconds,
                time.perf_counter() - finalize_started,
            )

    @staticmethod
    def _copy_plan(plan: TensorCopyPlan) -> None:
        copy = plan.recorded_copy
        parameter = getattr(copy.layer, copy.param_name)
        destination = parameter.as_strided(copy.shape, copy.stride, copy.offset)
        value = apply_chain(plan.staging_tensor, plan.replay_ops)
        destination.copy_(value)
