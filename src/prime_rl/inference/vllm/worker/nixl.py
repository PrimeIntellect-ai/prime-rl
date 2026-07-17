"""vLLM worker extension for composed, sharded NIXL weight pulls."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from math import prod
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from modelexpress import p2p_pb2
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
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc
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

logger = init_logger("prime_rl.inference.vllm.worker.nixl")


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
class WeightTransferGroup:
    reload_layer: nn.Module | None
    copies: list[TensorCopyPlan]
    persistent_copies: list[TensorCopyPlan]
    pulls: list[tuple[Any, Any, list[int]]]

    @property
    def destination_names(self) -> set[str]:
        return {plan.recorded_copy.param_name for plan in self.copies}


@dataclass
class WeightTransferPlan:
    receive_arena: torch.Tensor
    groups: list[WeightTransferGroup]
    total_bytes: int


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
        from modelexpress.client import MxClient

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
        trainer_ref = self.mx_rendezvous.wait_for_peers(timeout=self.weight_transfer_timeout)[0]
        table = decode_table(self.mx_rendezvous.fetch(trainer_ref).nixl_metadata)
        layers, persistent = self._bake(table)
        plan = self._build_pull_plan(table, layers, persistent)
        self.mx_rendezvous.publish(nixl_metadata=self.nixl_agent.get_metadata())
        # Join the current generation directly. Publishing a transient READY
        # before the first pull would let the trainer mistake initialization
        # for a completed acknowledgement.
        self.mx_rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.weight_transfer_plan = plan
        logger.info(
            "NIXL plan baked in %.2fs: rank=%d, groups=%d, copies=%d, bytes=%d, pull_lists=%d",
            time.perf_counter() - started,
            self.mx_rendezvous.rank,
            len(layers),
            sum(len(group.copies) + len(group.persistent_copies) for group in plan.groups),
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
    ) -> WeightTransferPlan:
        tensors = {tensor.name: tensor for tensor in table.tensors}
        copies = [copy for layer in layers for copy in layer.copies] + persistent
        specifications: dict[int, TensorCopySpec] = {}
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

        persistent_by_layer: dict[int, list[RecordedCopy]] = defaultdict(list)
        for copy in persistent:
            persistent_by_layer[id(copy.layer)].append(copy)
        work: list[tuple[nn.Module | None, list[RecordedCopy]]] = []
        for layer in layers:
            layer_copies = layer.copies + persistent_by_layer.pop(id(layer.layer), [])
            work.append((layer.layer, layer_copies))
        work.extend((None, layer_copies) for layer_copies in persistent_by_layer.values())

        arena_elements = max(
            sum(prod(specifications[id(copy)].staging_shape) for copy in layer_copies) for _, layer_copies in work
        )
        with classic_cuda_alloc():
            receive_arena = torch.empty(arena_elements, dtype=torch.bfloat16, device=self.device)
        self.nixl_agent.register_tensor(receive_arena)

        agent_devices: dict[int, int] = {
            shard.agent: shard.device_id for tensor in table.tensors for shard in tensor.shards
        }
        total_bytes = 0
        peer_names: dict[int, str] = {}
        transfer_groups: list[WeightTransferGroup] = []

        for reload_layer, layer_copies in work:
            copy_plans: list[TensorCopyPlan] = []
            persistent_copy_plans: list[TensorCopyPlan] = []
            local_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            remote_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            cursor = 0

            for copy in layer_copies:
                specification = specifications[id(copy)]
                numel = prod(specification.staging_shape)
                staging_tensor = receive_arena.narrow(0, cursor, numel).view(specification.staging_shape)
                cursor += numel
                copy_plan = TensorCopyPlan(
                    recorded_copy=copy,
                    staging_tensor=staging_tensor,
                    replay_ops=specification.replay_ops,
                )
                (persistent_copy_plans if copy.persistent else copy_plans).append(copy_plan)

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
                    reload_layer=reload_layer,
                    copies=copy_plans,
                    persistent_copies=persistent_copy_plans,
                    pulls=pulls,
                )
            )

        logger.info(
            "NIXL receive arena uses %.2f GB for %d sequential groups",
            receive_arena.nbytes / 1e9,
            len(transfer_groups),
        )
        return WeightTransferPlan(
            receive_arena=receive_arena,
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
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            for transfer_group in plan.groups:
                for local, remote, indices in transfer_group.pulls:
                    handle = self.nixl_agent.post_read(local, indices, remote)
                    self.nixl_agent.wait(
                        handle,
                        context="weight pull",
                        timeout=self.weight_transfer_timeout,
                    )
                torch.cuda.synchronize(self.device)

                layer = transfer_group.reload_layer
                if layer is None:
                    for copy_plan in transfer_group.persistent_copies:
                        self._copy_plan(copy_plan)
                    torch.cuda.synchronize(self.device)
                    continue

                info = LAYERWISE_INFO[layer]
                materialize_layer(layer, info)
                # Match loader semantics for destinations with unwritten padding.
                destination_names = transfer_group.destination_names
                for name, tensor in get_layer_tensors(layer).items():
                    if name in destination_names and not tensor.is_meta:
                        tensor.zero_()
                for copy_plan in transfer_group.copies:
                    self._copy_plan(copy_plan)
                for copy_plan in transfer_group.persistent_copies:
                    self._copy_plan(copy_plan)

                if hasattr(layer, "_already_called_process_weights_after_loading"):
                    delattr(layer, "_already_called_process_weights_after_loading")
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    quant_method.process_weights_after_loading(layer)
                if info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(layer, info)
                info.reset()
                torch.cuda.synchronize(self.device)

            finalize_layerwise_reload(model, self.model_runner.model_config)

    @staticmethod
    def _copy_plan(plan: TensorCopyPlan) -> None:
        copy = plan.recorded_copy
        parameter = getattr(copy.layer, copy.param_name)
        destination = parameter.as_strided(copy.shape, copy.stride, copy.offset)
        value = apply_chain(plan.staging_tensor, plan.replay_ops)
        destination.copy_(value)
