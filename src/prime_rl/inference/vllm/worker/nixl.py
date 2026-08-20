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
from prime_rl.trainer.rl.broadcast.nixl.agent import MemDesc, NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.trainer.rl.broadcast.nixl.cuda_malloc_memory import (
    size_cuda_buffers,
    use_cuda_malloc_pool,
)
from prime_rl.trainer.rl.broadcast.nixl.graph import (
    Destination,
    OperationChain,
    RecordedCopy,
    TensorReplayPlan,
    WeightLoadRecorder,
    apply_chain,
    make_hf_lazy_weights,
    plan_tensor_replay,
)
from prime_rl.trainer.rl.broadcast.nixl.model_express import ModelExpressSession
from prime_rl.trainer.rl.broadcast.nixl.tensor_routing import route_sharded_tensor
from prime_rl.trainer.rl.broadcast.nixl.trainer_tensor_table import TrainerTensorTable

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")
_BUFFER_POLL_INTERVAL = 0.01


@dataclass
class TensorCopyPlan:
    recorded_copy: RecordedCopy
    staging_tensor: torch.Tensor
    replay_ops: OperationChain


@dataclass
class LayerWeightTransferPlan:
    reload_layer: nn.Module | None
    copies: list[TensorCopyPlan]
    persistent_copies: list[TensorCopyPlan]

    @property
    def destination_names(self) -> set[str]:
        return {plan.recorded_copy.destination_name for plan in self.copies}


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
        global_rank = rank_offset + self.device.index
        server_url = f"{host}:{port}"
        set_ucx_env_defaults()
        self.nixl_agent = NixlAgent(make_agent_name("inference", global_rank))
        self.model_express = ModelExpressSession(
            client=MxClient(server_url=server_url),
            role="inference",
            rank=global_rank,
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
    def initialize_transfer(self) -> WeightTransferPlan:
        if self.weight_transfer_plan is not None:
            return self.weight_transfer_plan

        trainer_ref = self.model_express.wait_for(
            "trainer",
            count=1,
            status=None,
            timeout=self.weight_transfer_timeout,
        )[0]
        table = TrainerTensorTable.decode(self.model_express.fetch(trainer_ref).nixl_metadata)
        copies = self.trace_weight_loads(table)
        plan = self.build_transfer_plan(table, copies)
        self.buffer_sessions = []
        for buffer_index in range(table.staging_buffer_count):
            session = ModelExpressSession(
                client=self.model_express.client,
                role="inference",
                rank=self.model_express.rank,
                session_id=f"{self.model_express.session_id}:layers:{buffer_index}",
                worker_id=f"inference-buffer-{self.model_express.rank}-{buffer_index}",
            )
            session.publish()
            session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            self.buffer_sessions.append(session)
        # Join the current generation directly. Publishing a transient READY
        # before the first pull would let the trainer mistake initialization
        # for a completed acknowledgement.
        self.model_express.publish(nixl_metadata=self.nixl_agent.get_metadata())
        self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.weight_transfer_plan = plan
        logger.info(
            "Initialized NIXL transfer plan on rank %d with %d groups",
            self.model_express.rank,
            len(plan.groups),
        )
        return plan

    def trace_weight_loads(
        self,
        table: TrainerTensorTable,
    ) -> list[RecordedCopy]:
        """Trace vLLM weight loading into source-to-destination copies."""
        from vllm.model_executor.model_loader.reload.layerwise import (
            _get_original_loader,
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        model = self.raw_model
        recorder = WeightLoadRecorder()
        regular_by_layer: dict[int, list[RecordedCopy]] = defaultdict(list)
        persistent: list[RecordedCopy] = []
        original_loaders: list[tuple[torch.Tensor, Any]] = []
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            try:
                for module in model.modules():
                    for name, tensor in get_layer_tensors(module).items():
                        destination = Destination(module, name, tensor)
                        if not tensor.is_meta:
                            recorder.register_destination_storage(destination)
                        loader = _get_original_loader(tensor)
                        original_loaders.append((tensor, loader))
                        tensor.weight_loader = self.wrap_weight_loader_for_recording(
                            recorder,
                            destination,
                            loader,
                        )

                model.load_weights(
                    make_hf_lazy_weights(
                        table,
                        device=self.device,
                        recorder=recorder,
                        hf_config=self.model_runner.model_config.hf_text_config,
                    )
                )

                for copy in recorder.copies:
                    if copy.is_persistent or copy.destination_name in SKIP_TENSORS:
                        copy.is_persistent = True
                        persistent.append(copy)
                    else:
                        regular_by_layer[id(copy.destination_module)].append(copy)
            finally:
                try:
                    for tensor, loader in reversed(original_loaders):
                        tensor.weight_loader = loader
                finally:
                    self._restore_layerwise_state(model)

        regular = [copy for copies in regular_by_layer.values() for copy in copies]
        return regular + persistent

    @staticmethod
    def wrap_weight_loader_for_recording(
        recorder: WeightLoadRecorder,
        destination: Destination,
        loader: Any,
    ):
        @wraps(loader)
        def recording_loader(*args, **kwargs):
            recorder.active_destination = destination
            try:
                return loader(*args, **kwargs)
            finally:
                recorder.active_destination = None

        return recording_loader

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

    def build_transfer_plan(
        self,
        table: TrainerTensorTable,
        copies: list[RecordedCopy],
    ) -> WeightTransferPlan:
        replay_plans = self.plan_tensor_replays(table, copies)
        receive_buffer_elements = self.calculate_receive_buffer_elements(
            table,
            copies,
            replay_plans,
        )
        receive_buffer_count = self.choose_receive_buffer_count(
            receive_buffer_elements,
            table.staging_buffer_count,
        )
        receive_arenas = self.allocate_receive_arenas(
            receive_buffer_elements,
            receive_buffer_count,
        )
        groups = self.build_transfer_groups(
            table,
            copies,
            replay_plans,
            receive_buffer_elements,
            receive_arenas,
            receive_buffer_count,
        )
        return WeightTransferPlan(
            receive_arenas=receive_arenas,
            receive_buffer_count=receive_buffer_count,
            groups=groups,
        )

    def plan_tensor_replays(
        self,
        table: TrainerTensorTable,
        copies: list[RecordedCopy],
    ) -> dict[int, TensorReplayPlan]:
        tensors = {tensor.name: tensor for group in table.groups for tensor in group.tensors}
        replay_plans: dict[int, TensorReplayPlan] = {}
        for copy in copies:
            source = tensors[copy.source_name]
            replay_plans[id(copy)] = plan_tensor_replay(
                tuple(source.shape),
                getattr(torch, source.wire_dtype),
                copy.ops,
            )
        return replay_plans

    def calculate_receive_buffer_elements(
        self,
        table: TrainerTensorTable,
        copies: list[RecordedCopy],
        replay_plans: dict[int, TensorReplayPlan],
    ) -> dict[torch.dtype, int]:
        tensors = {tensor.name: tensor for group in table.groups for tensor in group.tensors}
        tensor_groups = {
            tensor.name: group_index for group_index, group in enumerate(table.groups) for tensor in group.tensors
        }
        group_elements: dict[torch.dtype, list[int]] = defaultdict(lambda: [0] * len(table.groups))
        for copy in copies:
            source = tensors[copy.source_name]
            source_dtype = getattr(torch, source.wire_dtype)
            group_elements[source_dtype][tensor_groups[source.name]] += prod(replay_plans[id(copy)].source_shape)
        return {dtype: max(elements, default=0) for dtype, elements in group_elements.items()}

    def choose_receive_buffer_count(
        self,
        receive_buffer_elements: dict[torch.dtype, int],
        staging_buffer_count: int,
    ) -> int:
        receive_buffer_bytes = max(
            1,
            sum(elements * dtype.itemsize for dtype, elements in receive_buffer_elements.items()),
        )
        allocated_bytes = torch.cuda.memory_allocated(self.device)
        peak_growth_bytes = max(
            0,
            torch.cuda.max_memory_allocated(self.device) - allocated_bytes,
        )
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        max_receive_buffers = min(2, staging_buffer_count) if peak_growth_bytes else 1
        if peak_growth_bytes or free_bytes < receive_buffer_bytes:
            torch.cuda.empty_cache()
        return size_cuda_buffers(
            receive_buffer_bytes,
            max_receive_buffers,
            self.device,
            extra_headroom_bytes=receive_buffer_bytes + peak_growth_bytes,
        )

    def allocate_receive_arenas(
        self,
        receive_buffer_elements: dict[torch.dtype, int],
        receive_buffer_count: int,
    ) -> dict[torch.dtype, torch.Tensor]:
        with use_cuda_malloc_pool():
            receive_arenas = {
                dtype: torch.empty(
                    receive_buffer_count * elements,
                    dtype=dtype,
                    device=self.device,
                )
                for dtype, elements in receive_buffer_elements.items()
                if elements
            }
        for arena in receive_arenas.values():
            self.nixl_agent.register_tensor(arena)
        return receive_arenas

    def build_transfer_groups(
        self,
        table: TrainerTensorTable,
        copies: list[RecordedCopy],
        replay_plans: dict[int, TensorReplayPlan],
        receive_buffer_elements: dict[torch.dtype, int],
        receive_arenas: dict[torch.dtype, torch.Tensor],
        receive_buffer_count: int,
    ) -> list[WeightTransferGroup]:
        tensors = {tensor.name: tensor for group in table.groups for tensor in group.tensors}
        tensor_groups = {
            tensor.name: group_index for group_index, group in enumerate(table.groups) for tensor in group.tensors
        }
        copies_by_group: dict[int, list[RecordedCopy]] = defaultdict(list)
        for copy in copies:
            copies_by_group[tensor_groups[copy.source_name]].append(copy)

        reload_layers: dict[int, nn.Module] = {}
        for copy in copies:
            if not copy.is_persistent:
                reload_layers.setdefault(id(copy.destination_module), copy.destination_module)

        reload_layer_groups: dict[int, int] = {}
        for copy in copies:
            layer_id = id(copy.destination_module)
            if layer_id not in reload_layers:
                continue
            source_group = tensor_groups[copy.source_name]
            previous_group = reload_layer_groups.setdefault(layer_id, source_group)
            if previous_group != source_group:
                raise RuntimeError(
                    f"vLLM reload layer {type(copy.destination_module).__name__} reads trainer groups "
                    f"{table.groups[previous_group].name!r} and {table.groups[source_group].name!r}"
                )

        agent_devices = {agent_index: agent.device_id for agent_index, agent in enumerate(table.agents)}
        peer_names: dict[int, str] = {}
        transfer_groups: list[WeightTransferGroup] = []

        for group_index, group in enumerate(table.groups):
            copy_plans_by_layer: dict[int, list[TensorCopyPlan]] = defaultdict(list)
            persistent_plans_by_layer: dict[int, list[TensorCopyPlan]] = defaultdict(list)
            local_descs: dict[int, list[MemDesc]] = defaultdict(list)
            remote_descs: dict[int, list[MemDesc]] = defaultdict(list)
            cursors = {
                dtype: (group_index % receive_buffer_count) * elements
                for dtype, elements in receive_buffer_elements.items()
            }

            for copy in copies_by_group[group_index]:
                replay_plan = replay_plans[id(copy)]
                source = tensors[copy.source_name]
                source_dtype = getattr(torch, source.wire_dtype)
                numel = prod(replay_plan.source_shape)
                cursor = cursors[source_dtype]
                staging_tensor = receive_arenas[source_dtype].narrow(0, cursor, numel).view(replay_plan.source_shape)
                cursors[source_dtype] += numel
                copy_plan = TensorCopyPlan(
                    recorded_copy=copy,
                    staging_tensor=staging_tensor,
                    replay_ops=replay_plan.replay_ops,
                )
                plans = persistent_plans_by_layer if copy.is_persistent else copy_plans_by_layer
                plans[id(copy.destination_module)].append(copy_plan)

                for route in route_sharded_tensor(replay_plan, source, staging_tensor):
                    local_descs[route.agent].append((route.destination_addr, route.nbytes, self.device.index))
                    remote_descs[route.agent].append((route.source_addr, route.nbytes, agent_devices[route.agent]))

            transfer_groups.append(
                WeightTransferGroup(
                    name=group.name,
                    layers=self.build_layer_transfer_plans(
                        reload_layers,
                        copy_plans_by_layer,
                        persistent_plans_by_layer,
                    ),
                    pulls=self.prepare_group_pulls(
                        table,
                        local_descs,
                        remote_descs,
                        peer_names,
                    ),
                )
            )
        return transfer_groups

    def build_layer_transfer_plans(
        self,
        reload_layers: dict[int, nn.Module],
        copy_plans_by_layer: dict[int, list[TensorCopyPlan]],
        persistent_plans_by_layer: dict[int, list[TensorCopyPlan]],
    ) -> list[LayerWeightTransferPlan]:
        layer_plans: list[LayerWeightTransferPlan] = []
        for layer_id, layer in reload_layers.items():
            copies = copy_plans_by_layer.get(layer_id, [])
            persistent_copies = persistent_plans_by_layer.get(layer_id, [])
            if copies:
                layer_plans.append(
                    LayerWeightTransferPlan(
                        reload_layer=layer,
                        copies=copies,
                        persistent_copies=persistent_copies,
                    )
                )
            elif persistent_copies:
                layer_plans.append(
                    LayerWeightTransferPlan(
                        reload_layer=None,
                        copies=[],
                        persistent_copies=persistent_copies,
                    )
                )

        remaining_persistent = [
            plan
            for layer_id, plans in persistent_plans_by_layer.items()
            if layer_id not in reload_layers
            for plan in plans
        ]
        if remaining_persistent:
            layer_plans.append(
                LayerWeightTransferPlan(
                    reload_layer=None,
                    copies=[],
                    persistent_copies=remaining_persistent,
                )
            )
        return layer_plans

    def prepare_group_pulls(
        self,
        table: TrainerTensorTable,
        local_descs: dict[int, list[MemDesc]],
        remote_descs: dict[int, list[MemDesc]],
        peer_names: dict[int, str],
    ) -> list[tuple[Any, Any, list[int]]]:
        pulls: list[tuple[Any, Any, list[int]]] = []
        for agent_index, remote in sorted(remote_descs.items()):
            peer_name = peer_names.get(agent_index)
            if peer_name is None:
                peer_name = self.nixl_agent.add_remote_agent(table.agents[agent_index].metadata)
                self.nixl_agent.make_connection(peer_name)
                peer_names[agent_index] = peer_name
            local_prepared = self.nixl_agent.prepare_xfer_dlist(local_descs[agent_index])
            remote_prepared = self.nixl_agent.prepare_xfer_dlist(remote, agent_name=peer_name)
            pulls.append((local_prepared, remote_prepared, list(range(len(remote)))))
        return pulls

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        del weight_dir
        plan = self.initialize_transfer()
        self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.model_express.wait_for(
            "trainer",
            count=1,
            status=p2p_pb2.SOURCE_STATUS_READY,
            timeout=self.weight_transfer_timeout,
        )

        started = time.perf_counter()
        self.apply_transfer_plan(plan)
        update_mla_absorbed_weights(self.raw_model)
        torch.cuda.synchronize(self.device)
        self.model_express.set_status(p2p_pb2.SOURCE_STATUS_READY)
        logger.info(
            "Applied NIXL policy update on rank %d in %.2fs",
            self.model_express.rank,
            time.perf_counter() - started,
        )

    def apply_transfer_plan(self, plan: WeightTransferPlan) -> None:
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

        def pull_group(group_index: int) -> WeightTransferGroup:
            transfer_group = plan.groups[group_index]
            session = self.buffer_sessions[group_index % len(self.buffer_sessions)]
            session.wait_for(
                "trainer",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.weight_transfer_timeout,
                poll_interval=_BUFFER_POLL_INTERVAL,
                cancelled=cancelled.is_set,
            )

            for local, remote, indices in transfer_group.pulls:
                handle = self.nixl_agent.post_read(local, indices, remote)
                self.nixl_agent.wait(
                    handle,
                    context=f"weight pull for {transfer_group.name}",
                    timeout=self.weight_transfer_timeout,
                    cancelled=cancelled.is_set,
                )
            return transfer_group

        def acknowledge_group(group_index: int) -> None:
            session = self.buffer_sessions[group_index % len(self.buffer_sessions)]
            session.set_status(p2p_pb2.SOURCE_STATUS_READY)
            session.wait_for(
                "trainer",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.weight_transfer_timeout,
                poll_interval=_BUFFER_POLL_INTERVAL,
                cancelled=cancelled.is_set,
            )
            session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)

        def prefetch_group(group_index: int) -> WeightTransferGroup:
            torch.cuda.set_device(self.device)
            transfer_group = pull_group(group_index)
            acknowledge_group(group_index)
            return transfer_group

        def replay_group(transfer_group: WeightTransferGroup) -> None:
            for layer_plan in transfer_group.layers:
                layer = layer_plan.reload_layer
                if layer is None:
                    for copy_plan in layer_plan.persistent_copies:
                        self.replay_tensor_copy(copy_plan)
                    continue

                info = LAYERWISE_INFO[layer]
                materialize_layer(layer, info)
                # Match loader semantics for destinations with unwritten padding.
                destination_names = layer_plan.destination_names
                for name, tensor in get_layer_tensors(layer).items():
                    if name in destination_names and not tensor.is_meta:
                        tensor.zero_()
                for copy_plan in layer_plan.copies:
                    self.replay_tensor_copy(copy_plan)
                for copy_plan in layer_plan.persistent_copies:
                    self.replay_tensor_copy(copy_plan)

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
            try:
                pull = executor.submit(prefetch_group, 0) if executor is not None else None
                for group_index in range(len(plan.groups)):
                    transfer_group = pull.result() if pull is not None else pull_group(group_index)

                    torch.cuda.synchronize(self.device)
                    if executor is not None and group_index + 1 < len(plan.groups):
                        pull = executor.submit(prefetch_group, group_index + 1)

                    replay_group(transfer_group)
                    torch.cuda.synchronize(self.device)

                    if not pipelined:
                        acknowledge_group(group_index)
            finally:
                cancelled.set()
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

            finalize_layerwise_reload(model, self.model_runner.model_config)

    @staticmethod
    def replay_tensor_copy(plan: TensorCopyPlan) -> None:
        copy = plan.recorded_copy
        parameter = getattr(copy.destination_module, copy.destination_name)
        destination = parameter.as_strided(
            copy.destination_shape,
            copy.destination_stride,
            copy.destination_offset,
        )
        value = apply_chain(plan.staging_tensor, plan.replay_ops)
        destination.copy_(value)
