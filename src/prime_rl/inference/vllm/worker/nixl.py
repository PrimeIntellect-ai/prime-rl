"""vLLM worker extension for composed, sharded NIXL weight pulls."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from math import prod
from typing import TYPE_CHECKING, Any

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
from prime_rl.weight_transfer.lazy import BakeRecorder, RecordedCopy, checkpoint_destinations
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
class _CopyPlan:
    copy: RecordedCopy
    stage: torch.Tensor
    replay_ops: OpChain


@dataclass
class _CopySpecification:
    copy: RecordedCopy
    transport_ops: OpChain
    replay_ops: OpChain
    stage_shape: tuple[int, ...]


@dataclass
class _BakedGroup:
    layer: nn.Module
    copies: list[RecordedCopy]

    @property
    def param_names(self) -> set[str]:
        return {copy.param_name for copy in self.copies}


@dataclass
class _TransferGroup:
    baked: _BakedGroup | None
    plans: list[_CopyPlan]
    persistent_plans: list[_CopyPlan]
    pull_specs: list[tuple[Any, Any, list[int]]]


class NIXLWeightUpdateWorker(Worker):
    @property
    def raw_model(self) -> nn.Module:
        model = self.model_runner.get_model()
        assert isinstance(model, nn.Module)
        return model

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
        if quantize_in_weight_transfer:
            raise NotImplementedError("NIXL weight transfer does not support quantized weights")
        if self.vllm_config.quant_config is not None:
            raise NotImplementedError("NIXL weight transfer currently supports only unquantized vLLM models")
        if self.vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("NIXL weight transfer does not yet support runtime EPLB remapping")
        if not session_id or session_id == "default":
            raise ValueError("NIXL weight transfer requires a run-unique, non-default session_id")

        local_rank = int(self.local_rank)
        global_rank = rank_offset + local_rank
        if not 0 <= global_rank < inference_world_size:
            raise ValueError(
                f"invalid NIXL inference rank: rank_offset={rank_offset}, local_rank={local_rank}, "
                f"global_rank={global_rank}, inference_world_size={inference_world_size}"
            )
        self._mx_url = f"{host}:{port}"
        self._global_rank = global_rank
        self._timeout = timeout
        self._session_id = session_id
        self._initialized = False
        logger.info(
            "NIXL worker configured: global_rank=%d, ModelExpress=%s, session=%s",
            self._global_rank,
            self._mx_url,
            self._session_id,
        )

    @torch.no_grad()
    def _lazy_init(self) -> None:
        if self._initialized:
            return
        from modelexpress.client import MxClient

        started = time.perf_counter()
        set_ucx_env_defaults()
        self.nixl_agent = NixlAgent(make_agent_name("inference", self._global_rank))
        self.rendezvous = MxRendezvous(
            client=MxClient(server_url=self._mx_url),
            role="inference",
            rank=self._global_rank,
            peer_world_size=1,
            session_id=self._session_id,
            worker_id=f"inference-{self._global_rank}",
        )
        trainer_ref = self.rendezvous.wait_for_peers(timeout=self._timeout)[0]
        self._table = decode_table(self.rendezvous.fetch(trainer_ref).nixl_metadata)
        self._groups, persistent = self._bake()
        self._transfer_groups = self._build_pull_plan(self._table, self._groups, persistent)
        self.rendezvous.publish(nixl_metadata=self.nixl_agent.get_metadata())
        # Join the current generation directly. Publishing a transient READY
        # before the first pull would let the trainer mistake initialization
        # for a completed acknowledgement.
        self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self._initialized = True
        logger.info(
            "NIXL plan baked in %.2fs: rank=%d, groups=%d, copies=%d, bytes=%d, pull_lists=%d",
            time.perf_counter() - started,
            self._global_rank,
            len(self._groups),
            sum(len(group.plans) + len(group.persistent_plans) for group in self._transfer_groups),
            self._total_pull_bytes,
            sum(len(group.pull_specs) for group in self._transfer_groups),
        )

    def _bake(self) -> tuple[list[_BakedGroup], list[RecordedCopy]]:
        from vllm.model_executor.model_loader.reload.layerwise import (
            _get_original_loader,
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        model = self.raw_model
        recorder = BakeRecorder()
        groups: list[_BakedGroup] = []
        persistent: list[RecordedCopy] = []
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            expected_destinations = checkpoint_destinations(model, skip_names=SKIP_TENSORS)
            initialize_layerwise_reload(model)
            try:
                for module in model.modules():
                    for name, tensor in get_layer_tensors(module).items():
                        if not tensor.is_meta:
                            recorder.register_live_destination(module, name, tensor)
                        tensor.weight_loader = self._stamp(recorder, module, name, _get_original_loader(tensor))

                model.load_weights(
                    make_hf_lazy_weights(
                        self._table,
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

                self._validate_persistent_coverage(persistent)
                recorder.validate_expected_destinations(expected_destinations)

                for layer_id, copies in by_layer.items():
                    layer = layers[layer_id]
                    expected = {name for name in get_layer_tensors(layer) if name not in SKIP_TENSORS}
                    recorded = {copy.param_name for copy in copies}
                    if expected != recorded:
                        raise RuntimeError(
                            f"incomplete lazy load for {type(layer).__name__}: "
                            f"expected destinations={sorted(expected)}, recorded={sorted(recorded)}"
                        )
                    self._validate_destination_coverage(layer, copies)
                    groups.append(_BakedGroup(layer=layer, copies=copies))
            finally:
                try:
                    self._restore_layerwise_state(model)
                finally:
                    self._remove_stamps(model)

        if not groups:
            raise RuntimeError("vLLM lazy bake recorded no loadable destinations")
        return groups, persistent

    @staticmethod
    def _stamp(recorder: BakeRecorder, layer: nn.Module, name: str, loader: Any):
        @wraps(loader)
        def stamped(*args, **kwargs):
            recorder.current = (layer, name)
            try:
                return loader(*args, **kwargs)
            finally:
                recorder.current = None

        stamped._prime_nixl_stamp_inner = loader
        return stamped

    @staticmethod
    def _remove_stamps(model: nn.Module) -> None:
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        for layer in model.modules():
            for tensor in get_layer_tensors(layer).values():
                loader = getattr(tensor, "weight_loader", None)
                while loader is not None and hasattr(loader, "_prime_nixl_stamp_inner"):
                    loader = loader._prime_nixl_stamp_inner
                if loader is not None:
                    tensor.weight_loader = loader

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

    @staticmethod
    def _validate_destination_coverage(layer: nn.Module, copies: list[RecordedCopy]) -> None:
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from vllm.model_executor.model_loader.reload.meta import SKIP_TENSORS
        from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

        copies_by_name: dict[str, list[RecordedCopy]] = defaultdict(list)
        for copy in copies:
            copies_by_name[copy.param_name].append(copy)

        for name, tensor in get_layer_tensors(layer).items():
            if name in SKIP_TENSORS:
                continue
            expected_numel = tensor.numel()
            if isinstance(layer, VocabParallelEmbedding):
                if getattr(layer, "num_added_embeddings", 0) != 0:
                    raise RuntimeError("NIXL reload does not support added-vocabulary embedding rows")
                rows = layer.shard_indices.num_org_elements
                expected_numel = rows * prod(tensor.shape[1:])
            NIXLWeightUpdateWorker._validate_tensor_copy_coverage(
                layer,
                name,
                tensor,
                copies_by_name[name],
                expected_numel=expected_numel,
            )

    @staticmethod
    def _validate_persistent_coverage(copies: list[RecordedCopy]) -> None:
        copies_by_destination: dict[tuple[int, str], list[RecordedCopy]] = defaultdict(list)
        layers: dict[int, nn.Module] = {}
        for copy in copies:
            layers[id(copy.layer)] = copy.layer
            copies_by_destination[(id(copy.layer), copy.param_name)].append(copy)

        for (layer_id, name), destination_copies in copies_by_destination.items():
            layer = layers[layer_id]
            tensor = getattr(layer, name)
            NIXLWeightUpdateWorker._validate_tensor_copy_coverage(
                layer,
                name,
                tensor,
                destination_copies,
                expected_numel=tensor.numel(),
            )

    @staticmethod
    def _validate_tensor_copy_coverage(
        layer: nn.Module,
        name: str,
        tensor: torch.Tensor,
        copies: list[RecordedCopy],
        *,
        expected_numel: int,
    ) -> None:
        label = f"{type(layer).__name__}.{name}"
        if tensor.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(f"unsupported NIXL destination dtype for {label}: {tensor.dtype}")
        if not tensor.is_contiguous():
            raise RuntimeError(
                f"NIXL destination coverage requires a contiguous tensor, got {label} stride={tuple(tensor.stride())}"
            )
        if not 0 <= expected_numel <= tensor.numel():
            raise RuntimeError(
                f"invalid expected NIXL coverage for {label}: expected={expected_numel}, size={tensor.numel()}"
            )

        storage_start = tensor.storage_offset()
        storage_end = storage_start + tensor.numel()
        expected_end = storage_start + expected_numel
        runs = sorted(run for copy in copies for run in region_elem_runs(copy.offset, copy.shape, copy.stride))
        cursor = storage_start
        for start, length in runs:
            end = start + length
            if start < storage_start or end > storage_end:
                raise RuntimeError(
                    f"lazy copy for {label} falls outside destination storage: "
                    f"run=({start}, {length}), storage=[{storage_start}, {storage_end})"
                )
            if start < cursor:
                raise RuntimeError(f"overlapping lazy copies for {label} at element {start}")
            if start > cursor:
                raise RuntimeError(
                    f"incomplete lazy copies for {label}: uncovered destination range [{cursor}, {start})"
                )
            cursor = end
        if cursor != expected_end:
            raise RuntimeError(
                f"incomplete lazy copies for {label}: covered through element {cursor}, expected {expected_end}"
            )

    def _build_pull_plan(
        self,
        table: TrainerTable,
        groups: list[_BakedGroup],
        persistent: list[RecordedCopy],
    ) -> list[_TransferGroup]:
        tensors = {tensor.name: tensor for tensor in table.tensors}
        copies = [copy for group in groups for copy in group.copies] + persistent
        specifications: dict[int, _CopySpecification] = {}
        for copy in copies:
            source = tensors.get(copy.src_name)
            if source is None:
                raise RuntimeError(f"lazy graph references missing trainer tensor {copy.src_name!r}")
            source_dtype = getattr(torch, source.dtype)
            if source.master_dtype != "float32" or source_dtype != torch.bfloat16:
                raise RuntimeError(
                    f"{source.name}: expected FP32 master and BF16 wire, got "
                    f"master={source.master_dtype}, wire={source.dtype}"
                )
            transport_ops, replay_ops, stage_shape, stage_dtype = split_transport_chain(
                tuple(source.shape), source_dtype, copy.ops
            )
            if stage_dtype != torch.bfloat16:
                raise RuntimeError(
                    f"transport prefix for {source.name!r} changed dtype to {stage_dtype}; casts must replay locally"
                )
            if copy.destination_dtype not in (torch.bfloat16, torch.float32):
                raise RuntimeError(
                    f"unsupported NIXL destination dtype for {copy.src_name!r}: {copy.destination_dtype}"
                )
            replayed = apply_chain(torch.empty(stage_shape, dtype=stage_dtype, device="meta"), replay_ops)
            if tuple(replayed.shape) != copy.shape:
                raise RuntimeError(
                    f"lazy replay shape mismatch for {copy.src_name!r}: "
                    f"replayed={tuple(replayed.shape)}, destination={copy.shape}"
                )
            if replayed.dtype not in (torch.bfloat16, torch.float32):
                raise RuntimeError(f"unsupported NIXL replay dtype for {copy.src_name!r}: {replayed.dtype}")
            specifications[id(copy)] = _CopySpecification(
                copy=copy,
                transport_ops=transport_ops,
                replay_ops=replay_ops,
                stage_shape=stage_shape,
            )

        persistent_by_layer: dict[int, list[RecordedCopy]] = defaultdict(list)
        for copy in persistent:
            persistent_by_layer[id(copy.layer)].append(copy)
        work: list[tuple[_BakedGroup | None, list[RecordedCopy]]] = []
        for group in groups:
            work.append((group, group.copies + persistent_by_layer.pop(id(group.layer), [])))
        work.extend((None, layer_copies) for layer_copies in persistent_by_layer.values())

        arena_elements = max(
            (sum(prod(specifications[id(copy)].stage_shape) for copy in group_copies) for _, group_copies in work),
            default=0,
        )
        if arena_elements <= 0:
            raise RuntimeError("NIXL pull plan has no receive elements")

        with classic_cuda_alloc():
            self._receive_arena = torch.empty(arena_elements, dtype=torch.bfloat16, device=self.device)
        self.nixl_agent.register_tensor(self._receive_arena)

        agent_devices: dict[int, int] = {
            shard.agent: shard.device_id for tensor in table.tensors for shard in tensor.shards
        }
        self._total_pull_bytes = 0
        peer_names: dict[int, str] = {}
        transfer_groups: list[_TransferGroup] = []

        for baked, group_copies in work:
            plans: list[_CopyPlan] = []
            persistent_plans: list[_CopyPlan] = []
            local_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            remote_descs: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            cursor = 0

            for copy in group_copies:
                specification = specifications[id(copy)]
                numel = prod(specification.stage_shape)
                stage = self._receive_arena.narrow(0, cursor, numel).view(specification.stage_shape)
                cursor += numel
                plan = _CopyPlan(copy=copy, stage=stage, replay_ops=specification.replay_ops)
                (persistent_plans if copy.persistent else plans).append(plan)

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
                for agent, source_addr, destination_addr, nbytes in zip_src_dst(source_pieces, tensor_runs(stage)):
                    local_descs[agent].append((destination_addr, nbytes, self.device.index))
                    remote_descs[agent].append((source_addr, nbytes, agent_devices[agent]))
                    self._total_pull_bytes += nbytes

            pull_specs: list[tuple[Any, Any, list[int]]] = []
            for agent_index, remote in sorted(remote_descs.items()):
                peer_name = peer_names.get(agent_index)
                if peer_name is None:
                    peer_name = self.nixl_agent.add_remote_agent(table.agents[agent_index].metadata)
                    self.nixl_agent.make_connection(peer_name)
                    peer_names[agent_index] = peer_name
                local_prepared = self.nixl_agent.prep_local(local_descs[agent_index])
                remote_prepared = self.nixl_agent.prep_remote(peer_name, remote)
                pull_specs.append((local_prepared, remote_prepared, list(range(len(remote)))))
            transfer_groups.append(
                _TransferGroup(
                    baked=baked,
                    plans=plans,
                    persistent_plans=persistent_plans,
                    pull_specs=pull_specs,
                )
            )

        logger.info(
            "NIXL receive arena uses %.2f GB for %d sequential groups",
            self._receive_arena.nbytes / 1e9,
            len(transfer_groups),
        )
        return transfer_groups

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        del weight_dir
        self._lazy_init()
        self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.rendezvous.wait_for(
            "trainer",
            count=1,
            status=p2p_pb2.SOURCE_STATUS_READY,
            timeout=self._timeout,
        )

        started = time.perf_counter()
        self._process_and_commit()
        update_mla_absorbed_weights(self.raw_model)
        torch.cuda.synchronize(self.device)
        self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
        logger.info(
            "Applied %.2f GB NIXL policy update on rank %d in %.2fs",
            self._total_pull_bytes / 1e9,
            self._global_rank,
            time.perf_counter() - started,
        )

    def _process_and_commit(self) -> None:
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
            try:
                for transfer_group in self._transfer_groups:
                    # The arena is reused by every group, so all reads and
                    # scatters for this group must finish before the next pull.
                    for local, remote, indices in transfer_group.pull_specs:
                        handle = self.nixl_agent.post_read(local, indices, remote)
                        self.nixl_agent.wait(handle, context="weight pull", timeout=self._timeout)
                    torch.cuda.synchronize(self.device)

                    group = transfer_group.baked
                    if group is None:
                        # A persistent-only destination has no materialized
                        # load-time parameters or post-processing step.
                        for plan in transfer_group.persistent_plans:
                            self._copy_plan(plan)
                        torch.cuda.synchronize(self.device)
                        continue

                    info = LAYERWISE_INFO[group.layer]
                    materialize_layer(group.layer, info)
                    # Vocab loaders explicitly zero their trailing padding.
                    # Zeroing every loaded destination also makes any allowed
                    # padding deterministic before the recorded prefix copies.
                    for name, tensor in get_layer_tensors(group.layer).items():
                        if name in group.param_names and not tensor.is_meta:
                            tensor.zero_()
                    for plan in transfer_group.plans:
                        self._copy_plan(plan)

                    # SKIP_TENSORS remain live across layerwise reload. Load
                    # them before post-processing so an upcast/downcast or
                    # repack observes the new generation.
                    for plan in transfer_group.persistent_plans:
                        self._copy_plan(plan)

                    if hasattr(group.layer, "_already_called_process_weights_after_loading"):
                        delattr(group.layer, "_already_called_process_weights_after_loading")
                    quant_method = getattr(group.layer, "quant_method", None)
                    if isinstance(quant_method, QuantizeMethodBase):
                        quant_method.process_weights_after_loading(group.layer)
                    if info.kernel_tensors is not None:
                        _copy_and_restore_kernel_tensors(group.layer, info)
                    info.reset()
                    # Deferred casts and kernel conversion can read the shared
                    # arena asynchronously. Finish them before RDMA reuses it.
                    torch.cuda.synchronize(self.device)

                finalize_layerwise_reload(model, self.model_runner.model_config)
            except Exception:
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    logger.exception("Failed to synchronize CUDA while aborting a NIXL update")
                try:
                    self._restore_layerwise_state(model)
                except Exception:
                    logger.exception("Failed to restore vLLM layerwise state after NIXL update failure")
                raise

    @staticmethod
    def _copy_plan(plan: _CopyPlan) -> None:
        copy = plan.copy
        parameter = getattr(copy.layer, copy.param_name)
        if parameter.dtype != copy.destination_dtype or parameter.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(
                f"NIXL destination dtype changed for {type(copy.layer).__name__}.{copy.param_name}: "
                f"baked={copy.destination_dtype}, runtime={parameter.dtype}"
            )
        destination = parameter.as_strided(copy.shape, copy.stride, copy.offset)
        value = apply_chain(plan.stage, plan.replay_ops)
        if tuple(value.shape) != copy.shape or value.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(
                f"invalid NIXL replay for {copy.src_name!r}: "
                f"shape={tuple(value.shape)}, dtype={value.dtype}, destination_shape={copy.shape}"
            )
        destination.copy_(value)
