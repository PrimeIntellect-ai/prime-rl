"""vLLM worker extension for composed, sharded NIXL weight pulls."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
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
from prime_rl.weight_transfer.wire import TrainerTable, decode_table

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")
_BUFFER_POLL_INTERVAL = 0.01
_MAX_INFLIGHT_READS = 1


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
    receive_arena: torch.Tensor
    receive_buffer_count: int
    groups: list[WeightTransferGroup]
    total_bytes: int


@dataclass
class PulledWeightTransferGroup:
    group: WeightTransferGroup
    source_wait_seconds: float
    transfer_seconds: float
    post_seconds: float
    handle_wait_seconds: float
    ready_publish_seconds: float
    slot_stall_seconds: float
    slot_recycle_seconds: float
    recycled_credit_seconds: float
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
        del quantize_in_weight_transfer
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
        self.inference_world_size = inference_world_size
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
        table = decode_table(self.mx_rendezvous.fetch(trainer_ref).nixl_metadata)
        layers, persistent = self._bake(table)
        plan = self._build_pull_plan(table, layers, persistent, allocated_bytes, peak_allocated_bytes)
        self.buffer_rendezvous = []
        for buffer_index in range(table.buffer_count):
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
            table.buffer_count,
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
        tensors = {tensor.name: tensor for tensor in table.tensors}
        copies = [copy for layer in layers for copy in layer.copies] + persistent
        specifications: dict[int, TensorCopySpec] = {}
        copies_by_group: dict[int, list[RecordedCopy]] = defaultdict(list)
        for copy in copies:
            source = tensors[copy.src_name]
            source_dtype = getattr(torch, source.dtype)
            transport_ops, replay_ops, staging_shape = split_transport_chain(
                tuple(source.shape), source_dtype, copy.ops
            )
            specifications[id(copy)] = TensorCopySpec(
                transport_ops=transport_ops,
                replay_ops=replay_ops,
                staging_shape=staging_shape,
            )
            copies_by_group[source.group].append(copy)

        reload_layer_ids = {id(layer.layer) for layer in layers}
        reload_layer_groups: dict[int, int] = {}
        for copy in copies:
            layer_id = id(copy.layer)
            if layer_id not in reload_layer_ids:
                continue
            source_group = tensors[copy.src_name].group
            previous_group = reload_layer_groups.setdefault(layer_id, source_group)
            if previous_group != source_group:
                raise RuntimeError(
                    f"vLLM reload layer {type(copy.layer).__name__} reads trainer groups "
                    f"{table.groups[previous_group]!r} and {table.groups[source_group]!r}"
                )

        largest_group_elements = max(
            (
                sum(prod(specifications[id(copy)].staging_shape) for copy in copies_by_group[group_index])
                for group_index in range(len(table.groups))
            ),
            default=0,
        )
        largest_group_elements = max(1, largest_group_elements)
        largest_group_bytes = largest_group_elements * torch.bfloat16.itemsize
        peak_growth_bytes = max(0, peak_allocated_bytes - allocated_bytes)
        has_observed_peak_growth = peak_growth_bytes > 0
        free_before_reclaim, _ = torch.cuda.mem_get_info(self.device)
        max_receive_buffers = min(2, table.buffer_count) if has_observed_peak_growth else 1
        if has_observed_peak_growth or free_before_reclaim < largest_group_bytes:
            torch.cuda.empty_cache()
        receive_buffer_count, free_bytes, device_total_bytes, headroom_bytes = cuda_buffer_capacity(
            largest_group_bytes,
            max_receive_buffers,
            self.device,
            extra_headroom_bytes=max(largest_group_bytes, peak_growth_bytes),
        )
        receive_arena_elements = receive_buffer_count * largest_group_elements
        with classic_cuda_alloc():
            receive_arena = torch.empty(
                receive_arena_elements,
                dtype=torch.bfloat16,
                device=self.device,
            )
        self.nixl_agent.register_tensor(receive_arena)
        post_free_bytes, _ = torch.cuda.mem_get_info(self.device)

        agent_devices: dict[int, int] = {
            shard.agent: shard.device_id for tensor in table.tensors for shard in tensor.shards
        }
        total_bytes = 0
        peer_names: dict[int, str] = {}
        transfer_groups: list[WeightTransferGroup] = []

        for group_index, group_name in enumerate(table.groups):
            copy_plans_by_layer: dict[int, list[TensorCopyPlan]] = defaultdict(list)
            persistent_plans_by_layer: dict[int, list[TensorCopyPlan]] = defaultdict(list)
            local_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            remote_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            cursor = (group_index % receive_buffer_count) * largest_group_elements

            for copy in copies_by_group[group_index]:
                specification = specifications[id(copy)]
                numel = prod(specification.staging_shape)
                staging_tensor = receive_arena.narrow(0, cursor, numel).view(specification.staging_shape)
                cursor += numel
                copy_plan = TensorCopyPlan(
                    recorded_copy=copy,
                    staging_tensor=staging_tensor,
                    replay_ops=specification.replay_ops,
                )
                plans = persistent_plans_by_layer if copy.persistent else copy_plans_by_layer
                plans[id(copy.layer)].append(copy_plan)

                source = tensors[copy.src_name]
                source_dtype = getattr(torch, source.dtype)
                offset, shape, stride = resolve_chain_region(
                    tuple(source.shape), source_dtype, specification.transport_ops
                )
                row_numel = prod(source.shape[1:]) if len(source.shape) > 1 else 1
                source_pieces = route_region(
                    region_elem_runs(offset, shape, stride),
                    source.shards,
                    row_numel,
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
            if receive_buffer_count > 1 and pulls:
                offset = (group_index + self.mx_rendezvous.rank * len(pulls) // self.inference_world_size) % len(pulls)
                pulls = pulls[offset:] + pulls[:offset]
            transfer_groups.append(
                WeightTransferGroup(
                    name=group_name,
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
            table.buffer_count,
            max_receive_buffers,
            free_before_reclaim / 1024**3,
            free_bytes / 1024**3,
            allocated_bytes / 1024**3,
            peak_allocated_bytes / 1024**3,
            peak_growth_bytes / 1024**3,
            device_total_bytes / 1024**3,
            headroom_bytes / 1024**3,
            largest_group_bytes / 1024**3,
            receive_arena.nbytes / 1024**3,
            post_free_bytes / 1024**3,
            len(transfer_groups),
        )
        return WeightTransferPlan(
            receive_arena=receive_arena,
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
        source_buffer_count = len(self.buffer_rendezvous)
        slot_recycles: list[tuple[int, Future[tuple[float, float]]] | None] = [None] * source_buffer_count

        def await_slot_recycle(group_index: int) -> tuple[float, float, float]:
            slot = group_index % source_buffer_count
            pending = slot_recycles[slot]
            if pending is None:
                return 0.0, 0.0, 0.0
            started = time.perf_counter()
            recycle_seconds, credit_seconds = pending[1].result()
            stall_seconds = time.perf_counter() - started
            slot_recycles[slot] = None
            return stall_seconds, recycle_seconds, credit_seconds

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
            post_seconds = 0.0
            handle_wait_seconds = 0.0
            if plan.receive_buffer_count == 1:
                for local, remote, indices in transfer_group.pulls:
                    handle = self.nixl_agent.post_read(local, indices, remote)
                    self.nixl_agent.wait(
                        handle,
                        context=f"weight pull for {transfer_group.name}",
                        timeout=self.weight_transfer_timeout,
                        cancelled=cancelled.is_set,
                    )
            else:
                handles: deque[Any] = deque()
                for local, remote, indices in transfer_group.pulls:
                    post_started = time.perf_counter()
                    handles.append(self.nixl_agent.post_read(local, indices, remote))
                    post_seconds += time.perf_counter() - post_started
                    if len(handles) == _MAX_INFLIGHT_READS:
                        wait_started = time.perf_counter()
                        self.nixl_agent.wait(
                            handles.popleft(),
                            context=f"weight pull for {transfer_group.name}",
                            timeout=self.weight_transfer_timeout,
                            cancelled=cancelled.is_set,
                        )
                        handle_wait_seconds += time.perf_counter() - wait_started
                while handles:
                    wait_started = time.perf_counter()
                    self.nixl_agent.wait(
                        handles.popleft(),
                        context=f"weight pull for {transfer_group.name}",
                        timeout=self.weight_transfer_timeout,
                        cancelled=cancelled.is_set,
                    )
                    handle_wait_seconds += time.perf_counter() - wait_started
            transfer_seconds = time.perf_counter() - transfer_started

            return PulledWeightTransferGroup(
                group=transfer_group,
                source_wait_seconds=source_wait_seconds,
                transfer_seconds=transfer_seconds,
                post_seconds=post_seconds,
                handle_wait_seconds=handle_wait_seconds,
                ready_publish_seconds=0.0,
                slot_stall_seconds=0.0,
                slot_recycle_seconds=0.0,
                recycled_credit_seconds=0.0,
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

        def recycle_slot(group_index: int, ready_at: float) -> tuple[float, float]:
            rendezvous = self.buffer_rendezvous[group_index % source_buffer_count]
            started = time.perf_counter()
            rendezvous.wait_for(
                "trainer",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.weight_transfer_timeout,
                poll_interval=_BUFFER_POLL_INTERVAL,
                cancelled=cancelled.is_set,
            )
            rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            finished = time.perf_counter()
            return finished - started, finished - ready_at

        def prefetch_group(
            group_index: int,
            recycle_executor: ThreadPoolExecutor,
        ) -> PulledWeightTransferGroup:
            torch.cuda.set_device(self.device)
            slot_stall_seconds, slot_recycle_seconds, recycled_credit_seconds = await_slot_recycle(group_index)
            pulled = pull_group(group_index)
            pulled.slot_stall_seconds = slot_stall_seconds
            pulled.slot_recycle_seconds = slot_recycle_seconds
            pulled.recycled_credit_seconds = recycled_credit_seconds
            ready_started = time.perf_counter()
            rendezvous = self.buffer_rendezvous[group_index % source_buffer_count]
            rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
            ready_at = time.perf_counter()
            pulled.ready_publish_seconds = ready_at - ready_started
            slot_recycles[group_index % source_buffer_count] = (
                group_index,
                recycle_executor.submit(recycle_slot, group_index, ready_at),
            )
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
            initialize_started = time.perf_counter()
            initialize_layerwise_reload(model)
            initialize_seconds = time.perf_counter() - initialize_started
            pipelined = plan.receive_buffer_count > 1
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nixl-prefetch") if pipelined else None
            recycle_executor = (
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="nixl-credit") if pipelined else None
            )
            source_wait_seconds = 0.0
            transfer_seconds = 0.0
            post_seconds = 0.0
            handle_wait_seconds = 0.0
            ready_publish_seconds = 0.0
            slot_stall_seconds = 0.0
            slot_recycle_seconds = 0.0
            recycled_credit_seconds: list[float] = []
            initial_pull_wait_seconds = 0.0
            exposed_prefetch_wait_seconds = 0.0
            final_drain_seconds = 0.0
            acknowledgement_seconds = 0.0
            replay_seconds = 0.0
            slowest_transfer_seconds = 0.0
            slowest_transfer_group = plan.groups[0].name
            pipeline_started = time.perf_counter()
            try:
                pull = (
                    executor.submit(prefetch_group, 0, recycle_executor)
                    if executor is not None and recycle_executor is not None
                    else None
                )
                for group_index in range(len(plan.groups)):
                    if pull is not None:
                        pull_wait_started = time.perf_counter()
                        pulled = pull.result()
                        pull_wait_seconds = time.perf_counter() - pull_wait_started
                        if group_index == 0:
                            initial_pull_wait_seconds = pull_wait_seconds
                        else:
                            exposed_prefetch_wait_seconds += pull_wait_seconds
                    else:
                        pulled = pull_group(group_index)
                    transfer_group = pulled.group
                    source_wait_seconds += pulled.source_wait_seconds
                    transfer_seconds += pulled.transfer_seconds
                    post_seconds += pulled.post_seconds
                    handle_wait_seconds += pulled.handle_wait_seconds
                    ready_publish_seconds += pulled.ready_publish_seconds
                    slot_stall_seconds += pulled.slot_stall_seconds
                    slot_recycle_seconds += pulled.slot_recycle_seconds
                    if pulled.recycled_credit_seconds:
                        recycled_credit_seconds.append(pulled.recycled_credit_seconds)
                    acknowledgement_seconds += pulled.acknowledgement_seconds
                    if pulled.transfer_seconds > slowest_transfer_seconds:
                        slowest_transfer_seconds = pulled.transfer_seconds
                        slowest_transfer_group = transfer_group.name

                    torch.cuda.synchronize(self.device)
                    if executor is not None and recycle_executor is not None and group_index + 1 < len(plan.groups):
                        pull = executor.submit(prefetch_group, group_index + 1, recycle_executor)

                    replay_started = time.perf_counter()
                    replay_group(transfer_group)
                    torch.cuda.synchronize(self.device)
                    replay_seconds += time.perf_counter() - replay_started

                    if not pipelined:
                        acknowledgement_seconds += acknowledge_group(group_index)

                if pipelined:
                    final_drain_started = time.perf_counter()
                    pending_groups = sorted(pending[0] for pending in slot_recycles if pending is not None)
                    for group_index in pending_groups:
                        _, recycle_seconds, credit_seconds = await_slot_recycle(group_index)
                        slot_recycle_seconds += recycle_seconds
                        if credit_seconds:
                            recycled_credit_seconds.append(credit_seconds)
                    final_drain_seconds = time.perf_counter() - final_drain_started
            finally:
                cancelled.set()
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                if recycle_executor is not None:
                    recycle_executor.shutdown(wait=True, cancel_futures=True)
            pipeline_seconds = time.perf_counter() - pipeline_started

            finalize_started = time.perf_counter()
            finalize_layerwise_reload(model, self.model_runner.model_config)
            finalize_seconds = time.perf_counter() - finalize_started
            if pipelined:
                average_credit_seconds = sum(recycled_credit_seconds) / max(len(recycled_credit_seconds), 1)
                maximum_credit_seconds = max(recycled_credit_seconds, default=0.0)
                logger.info(
                    "NIXL update profile rank=%d: groups=%d, source_wait=%.2fs, "
                    "rdma=%.2fs (post=%.2fs, wait=%.2fs, window=%d, %.1f Gb/s, "
                    "slowest=%s/%.2fs), sync=(ready_publish=%.2fs, credit_avg=%.2fs, "
                    "credit_max=%.2fs, recycle_worker_sum=%.2fs, slot_wait_sum=%.2fs, "
                    "final_drain=%.2fs), initial_pull_wait=%.2fs, exposed_prefetch_wait=%.2fs, "
                    "replay=%.2fs, initialize=%.2fs, pipeline=%.2fs, finalize=%.2fs",
                    self.mx_rendezvous.rank,
                    len(plan.groups),
                    source_wait_seconds,
                    transfer_seconds,
                    post_seconds,
                    handle_wait_seconds,
                    _MAX_INFLIGHT_READS,
                    plan.total_bytes * 8 / max(transfer_seconds, 1e-9) / 1e9,
                    slowest_transfer_group,
                    slowest_transfer_seconds,
                    ready_publish_seconds,
                    average_credit_seconds,
                    maximum_credit_seconds,
                    slot_recycle_seconds,
                    slot_stall_seconds,
                    final_drain_seconds,
                    initial_pull_wait_seconds,
                    exposed_prefetch_wait_seconds,
                    replay_seconds,
                    initialize_seconds,
                    pipeline_seconds,
                    finalize_seconds,
                )
            else:
                logger.info(
                    "NIXL update profile rank=%d: groups=%d, source_wait=%.2fs, "
                    "sequential_rdma=%.2fs, source_ack=%.2fs, replay=%.2fs, finalize=%.2fs",
                    self.mx_rendezvous.rank,
                    len(plan.groups),
                    source_wait_seconds,
                    transfer_seconds,
                    acknowledgement_seconds,
                    replay_seconds,
                    finalize_seconds,
                )

    @staticmethod
    def _copy_plan(plan: TensorCopyPlan) -> None:
        copy = plan.recorded_copy
        parameter = getattr(copy.layer, copy.param_name)
        destination = parameter.as_strided(copy.shape, copy.stride, copy.offset)
        value = apply_chain(plan.staging_tensor, plan.replay_ops)
        destination.copy_(value)
