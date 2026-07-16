"""vLLM worker extension for sharded, pull-based NIXL weight updates."""

from __future__ import annotations

import time
from collections import defaultdict
from math import prod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import update_mla_absorbed_weights
from prime_rl.weight_transfer.chains import region_elem_runs, resolve_chain_region
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc
from prime_rl.weight_transfer.lazy import BakeRecorder, LazyWeight, RecordedCopy
from prime_rl.weight_transfer.mx import MxChannel
from prime_rl.weight_transfer.nixl import NixlAgent, agent_name, configure_ucx
from prime_rl.weight_transfer.publication import route_published_region
from prime_rl.weight_transfer.sharding import zip_source_destination
from prime_rl.weight_transfer.wire import (
    SyncSignal,
    WeightManifest,
    decode_manifest,
    decode_signal,
    encode_signal,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported published dtype {name!r}")
    return dtype


class _BakedGroup:
    def __init__(self, layer: nn.Module, copies: list[RecordedCopy]) -> None:
        self.layer = layer
        self.copies = copies
        self.param_names = sorted({copy.param_name for copy in copies})
        self.arena_offsets: dict[int, int] = {}
        self.arena_dtypes: dict[int, torch.dtype] = {}
        self.pull_specs: list[tuple[Any, Any, list[int]]] = []
        self.num_bytes = 0


class NIXLWeightUpdateWorker(Worker):
    @property
    def raw_model(self) -> nn.Module:
        model = self.model_runner.model
        model = model.runnable if hasattr(model, "runnable") else model
        if not isinstance(model, nn.Module):
            raise TypeError(f"expected a torch module, got {type(model).__name__}")
        return model

    def liveness_probe(self) -> None:
        return None

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        server_world_size: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
        session_id: str = "",
        model_name: str = "",
    ) -> None:
        if self.vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("NIXL weight broadcast does not support EPLB")
        if quantize_in_weight_transfer:
            raise ValueError("NIXL uses vLLM layerwise processing; configure inference quantization instead")
        tp_size = self.parallel_config.tensor_parallel_size
        tp_rank = self.rank % tp_size
        if server_world_size == tp_size:
            relative_rank = tp_rank
        else:
            dp_rank = self.parallel_config.data_parallel_index
            candidate_rank = dp_rank * tp_size + tp_rank
            if rank_offset <= candidate_rank < rank_offset + server_world_size:
                relative_rank = candidate_rank - rank_offset
            elif candidate_rank < server_world_size:
                relative_rank = candidate_rank
            else:
                raise ValueError(
                    f"vLLM rank {candidate_rank} is outside this admin server's "
                    f"rank span [{rank_offset}, {rank_offset + server_world_size})"
                )
        self._rank = rank_offset + relative_rank
        self._world_size = inference_world_size
        self._timeout = timeout
        self._session_id = session_id
        self._model_name = model_name or self.model_runner.model_config.model
        self._mx = MxChannel(
            f"{host}:{port}",
            session_id,
            self._model_name,
            "inference",
            "sync",
            self._rank,
        )
        self._initialized = False
        logger.info(
            f"NIXL pull worker configured: rank={self._rank}, rank_offset={rank_offset}, "
            f"server_world_size={server_world_size}, session={session_id}"
        )

    @torch.no_grad()
    def _lazy_init(self) -> None:
        if self._initialized:
            return
        started = time.perf_counter()

        def valid_manifest(payload: bytes) -> bool:
            manifest = decode_manifest(payload)
            return manifest.session_id == self._session_id and manifest.model == self._model_name

        payloads = self._mx.wait_for("trainer", "manifest", 1, valid_manifest, self._timeout)
        self._manifest = decode_manifest(next(iter(payloads.values())))

        configure_ucx(self.device.index or 0)
        self._agent = NixlAgent(agent_name("inference", self._rank, f"{self._session_id}-{self._manifest.epoch}"))
        self._groups = self._bake(self._manifest)
        self._allocate_arena(self._manifest, self._groups)
        self._build_pull_plans(self._manifest, self._groups)
        self._initialized = True
        logger.info(
            f"NIXL pull plan baked in {time.perf_counter() - started:.2f}s: "
            f"rank={self._rank}, groups={len(self._groups)}, bytes={self._total_pull_bytes:,}"
        )

    def _bake(self, manifest: WeightManifest) -> list[_BakedGroup]:
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _get_original_loader,
            _place_kernel_tensors,
            initialize_layerwise_reload,
        )
        from vllm.model_executor.model_loader.reload.utils import get_layer_size, get_layer_tensors

        recorder = BakeRecorder()
        model = self.raw_model
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            for module in model.modules():
                for name, tensor in get_layer_tensors(module).items():
                    # Stamp tensors even when layerwise reload deliberately
                    # skipped wrapping them. Some models (for example
                    # Nemotron's e_score_correction_bias) still load those via
                    # vLLM's default loader, and the bake must attribute that
                    # copy to its destination like any custom loader copy.
                    tensor.weight_loader = self._stamp(recorder, module, name, _get_original_loader(tensor))

            weights = (
                (
                    tensor.name,
                    LazyWeight(
                        tensor.name,
                        torch.Size(tensor.shape),
                        _torch_dtype(tensor.dtype),
                        self.device,
                        recorder,
                    ),
                )
                for tensor in manifest.tensors
            )
            model.load_weights(weights)

            copies_by_layer: dict[int, list[RecordedCopy]] = defaultdict(list)
            layers: dict[int, nn.Module] = {}
            for copy in recorder.copies:
                copies_by_layer[id(copy.layer)].append(copy)
                layers[id(copy.layer)] = copy.layer

            groups: list[_BakedGroup] = []
            for layer_id, copies in copies_by_layer.items():
                layer = layers[layer_id]
                covered = sum(prod(copy.shape) for copy in copies)
                expected = get_layer_size(layer)
                if covered < expected:
                    raise RuntimeError(
                        f"NIXL bake covered {covered}/{expected} elements of {type(layer).__name__}; "
                        "partial reloads are not allowed"
                    )
                groups.append(_BakedGroup(layer, copies))

            self._param_layout: dict[tuple[int, str], tuple[torch.Size, torch.dtype]] = {}
            for group in groups:
                for name in group.param_names:
                    parameter = getattr(group.layer, name)
                    self._param_layout[(id(group.layer), name)] = (parameter.shape, parameter.dtype)

            for layer in model.modules():
                info = LAYERWISE_INFO.get(layer)
                if info is not None and info.can_load():
                    if info.kernel_tensors is not None:
                        _place_kernel_tensors(layer, info)
                    info.reset()
            if hasattr(model, "_original_do_torchao_reload"):
                model._do_torchao_reload = model._original_do_torchao_reload

        if not groups:
            raise RuntimeError("vLLM consumed no published weights during the NIXL bake")
        return groups

    @staticmethod
    def _stamp(recorder: BakeRecorder, layer: nn.Module, name: str, loader: Any):
        def stamped(*args, **kwargs):
            recorder.current = (layer, name)
            try:
                return loader(*args, **kwargs)
            finally:
                recorder.current = None

        return stamped

    def _allocate_arena(self, manifest: WeightManifest, groups: list[_BakedGroup]) -> None:
        source_dtypes = {tensor.name: _torch_dtype(tensor.dtype) for tensor in manifest.tensors}
        max_bytes = 0
        for group in groups:
            offset = 0
            for index, copy in enumerate(group.copies):
                dtype = source_dtypes[copy.src_name]
                offset = (offset + 255) // 256 * 256
                group.arena_offsets[index] = offset
                group.arena_dtypes[index] = dtype
                offset += prod(copy.shape) * dtype.itemsize
            group.num_bytes = offset
            max_bytes = max(max_bytes, offset)
        with classic_cuda_alloc():
            self._arena = torch.empty(max_bytes, dtype=torch.uint8, device=self.device)
        self._agent.register_tensor(self._arena)

    def _build_pull_plans(self, manifest: WeightManifest, groups: list[_BakedGroup]) -> None:
        tensors = {tensor.name: tensor for tensor in manifest.tensors}
        remote_names: dict[int, str] = {}
        for index, descriptor in enumerate(manifest.agents):
            remote_name = self._agent.add_remote_agent(descriptor.metadata)
            self._agent.connect(remote_name)
            remote_names[index] = remote_name

        agent_devices: dict[int, int] = {}
        for tensor in manifest.tensors:
            for segment in tensor.segments:
                previous = agent_devices.setdefault(segment.agent, segment.device_id)
                if previous != segment.device_id:
                    raise ValueError(f"trainer agent {segment.agent} published multiple CUDA device ids")

        self._total_pull_bytes = 0
        for group in groups:
            local: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            remote: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            for index, copy in enumerate(group.copies):
                tensor = tensors.get(copy.src_name)
                if tensor is None:
                    raise KeyError(f"vLLM requested unpublished HF tensor {copy.src_name!r}")
                source_dtype = _torch_dtype(tensor.dtype)
                offset, shape, stride = resolve_chain_region(tensor.shape, source_dtype, copy.ops)
                source = route_published_region(
                    tensor,
                    region_elem_runs(offset, shape, stride),
                    source_dtype.itemsize,
                )
                arena_address = self._arena.data_ptr() + group.arena_offsets[index]
                num_bytes = prod(copy.shape) * source_dtype.itemsize
                for agent, source_address, destination_address, length in zip_source_destination(
                    source,
                    [(arena_address, num_bytes)],
                ):
                    local[agent].append((destination_address, length, self.device.index or 0))
                    remote[agent].append((source_address, length, agent_devices[agent]))
                    self._total_pull_bytes += length

            for agent, descriptors in sorted(remote.items()):
                local_prepared = self._agent.prepare_local(local[agent])
                remote_prepared = self._agent.prepare_remote(remote_names[agent], descriptors)
                group.pull_specs.append((local_prepared, remote_prepared, list(range(len(descriptors)))))

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None, step: int = 0) -> None:
        self._lazy_init()
        manifest = self._manifest

        def ready(payload: bytes) -> bool:
            signal = decode_signal(payload)
            return (
                signal.session_id == manifest.session_id
                and signal.epoch == manifest.epoch
                and signal.step == step
                and signal.phase == "trainer_ready"
                and signal.fingerprint == manifest.fingerprint
            )

        self._mx.wait_for("trainer", "sync", 1, ready, self._timeout)
        started = time.perf_counter()
        self._reload_groups()
        update_mla_absorbed_weights(self.raw_model)
        torch.cuda.synchronize(self.device)
        self._mx.publish(
            encode_signal(
                SyncSignal(
                    session_id=manifest.session_id,
                    epoch=manifest.epoch,
                    step=step,
                    phase="inference_applied",
                    rank=self._rank,
                    fingerprint=manifest.fingerprint,
                )
            )
        )
        logger.info(
            f"NIXL weight update v{step}: {self._total_pull_bytes / 1e9:.2f} GB pulled and processed "
            f"in {time.perf_counter() - started:.2f}s"
        )

    def _reload_groups(self) -> None:
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        model = self.raw_model
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            for group in self._groups:
                handles = [
                    self._agent.read(local, indices, remote, indices) for local, remote, indices in group.pull_specs
                ]
                for handle in handles:
                    self._agent.wait(
                        handle,
                        context=f"{type(group.layer).__name__} weight pull",
                        timeout=self._timeout,
                    )

                for name in group.param_names:
                    shape, dtype = self._param_layout[(id(group.layer), name)]
                    setattr(group.layer, name, nn.Parameter(torch.empty(shape, dtype=dtype, device=self.device), False))
                for index, copy in enumerate(group.copies):
                    source_dtype = group.arena_dtypes[index]
                    num_bytes = prod(copy.shape) * source_dtype.itemsize
                    source = (
                        self._arena.narrow(0, group.arena_offsets[index], num_bytes).view(source_dtype).view(copy.shape)
                    )
                    destination = getattr(group.layer, copy.param_name).as_strided(
                        copy.shape,
                        copy.stride,
                        copy.offset,
                    )
                    destination.copy_(source)

                quant_method = getattr(group.layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    if hasattr(group.layer, "_already_called_process_weights_after_loading"):
                        delattr(group.layer, "_already_called_process_weights_after_loading")
                    quant_method.process_weights_after_loading(group.layer)
                info = LAYERWISE_INFO.get(group.layer)
                if info is not None and info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(group.layer, info)
                if info is not None:
                    info.reset()
            finalize_layerwise_reload(model, self.model_runner.model_config)
