"""vLLM worker extension for sharded, pull-based NIXL weight updates."""

from __future__ import annotations

import time
from collections import defaultdict
from inspect import getclosurevars
from math import prod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import update_mla_absorbed_weights
from prime_rl.weight_transfer.chains import region_elem_runs, resolve_chain_region, tensor_runs
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc
from prime_rl.weight_transfer.diagnostics import fingerprint_tensor
from prime_rl.weight_transfer.kernel_graph import KernelGraphRecorder, TensorMeta, encode_graph_value
from prime_rl.weight_transfer.lazy import BakeRecorder, LazyWeight, RecordedCopy
from prime_rl.weight_transfer.mx import MxChannel
from prime_rl.weight_transfer.nixl import NixlAgent, agent_name, configure_ucx
from prime_rl.weight_transfer.publication import route_published_region
from prime_rl.weight_transfer.sharding import zip_source_destination
from prime_rl.weight_transfer.wire import (
    KernelInput,
    KernelLayerPlan,
    KernelOutput,
    KernelPlan,
    KernelSourceCopy,
    SyncSignal,
    WeightManifest,
    decode_diagnostics,
    decode_kernel_buffers,
    decode_manifest,
    decode_signal,
    encode_kernel_plan,
    encode_signal,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nixl")

_DIAGNOSTIC_SAMPLES_PER_TENSOR = 256


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported published dtype {name!r}")
    return dtype


def _tensor_set_signature(
    module: nn.Module,
    names: tuple[str, ...] | list[str],
) -> tuple[int, int, float, float, float, float, float]:
    """Return a deterministic, bounded-cost signature for a set of tensors.

    This is intentionally a diagnostic rather than a cryptographic checksum.
    It samples every tensor, salts samples by tensor name, and includes several
    independent moments so update-to-update equality is very unlikely to hide
    a changed or permuted tensor. Only the final seven scalars synchronize to
    the host, regardless of how many tensors are represented.
    """

    device = None
    accumulator = None
    tensor_count = 0
    total_numel = 0
    for name in names:
        tensor = getattr(module, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        if device is None:
            device = tensor.device
            accumulator = torch.zeros(5, dtype=torch.float64, device=device)
        if tensor.device != device:
            raise RuntimeError("reload diagnostic tensor set spans multiple devices")

        flat = tensor.detach().reshape(-1)
        stride = max(flat.numel() // _DIAGNOSTIC_SAMPLES_PER_TENSOR, 1)
        sample = flat[::stride][:_DIAGNOSTIC_SAMPLES_PER_TENSOR].to(torch.float64)
        finite = torch.isfinite(sample)
        values = torch.where(finite, sample, torch.zeros_like(sample))
        salt = 1 + sum((index + 1) * ord(char) for index, char in enumerate(name)) % 997
        positions = torch.arange(1, values.numel() + 1, dtype=torch.float64, device=device)
        assert accumulator is not None
        accumulator[0] += values.sum() * salt
        accumulator[1] += values.abs().sum() * salt
        accumulator[2] += values.square().sum() * salt
        accumulator[3] += (values * positions).sum() * salt
        accumulator[4] += (~finite).sum()
        tensor_count += 1
        total_numel += tensor.numel()

    if accumulator is None:
        return (0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stats = accumulator.cpu().tolist()
    return (tensor_count, total_numel, *stats)


class _BakedGroup:
    def __init__(self, layer: nn.Module, copies: list[RecordedCopy]) -> None:
        self.layer = layer
        self.copies = copies
        self.param_names = sorted({copy.param_name for copy in copies})
        self.arena_offsets: dict[int, int] = {}
        self.arena_dtypes: dict[int, torch.dtype] = {}
        self.pull_specs: list[tuple[Any, Any, list[int]]] = []
        self.num_bytes = 0
        self.kernel_graph = None
        self.kernel_outputs: dict[str, TensorMeta] = {}


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
        validate_reload: bool = False,
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
        self._validate_reload = validate_reload
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
        self._publish_kernel_plan(self._manifest, self._groups)
        self._build_kernel_pulls(self._manifest, self._groups)
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
        self._layer_names = {id(module): name or "<root>" for name, module in model.named_modules()}
        self._post_load_transforms: dict[tuple[int, str], Any] = {}
        stamped_tensors: set[int] = set()
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            for module in model.modules():
                for name, tensor in get_layer_tensors(module).items():
                    # Shared/aliased parameters can be registered on more than
                    # one module. Stamping the same callable repeatedly hides
                    # loader identity checks in model-specific load_weights.
                    if id(tensor) in stamped_tensors:
                        continue
                    stamped_tensors.add(id(tensor))
                    # Stamp tensors even when layerwise reload deliberately
                    # skipped wrapping them. Some models (for example
                    # Nemotron's e_score_correction_bias) still load those via
                    # vLLM's default loader, and the bake must attribute that
                    # copy to its destination like any custom loader copy.
                    loader = _get_original_loader(tensor)
                    if loader.__name__ == "composed_loader":
                        transform = getclosurevars(loader).nonlocals.get("fn")
                        if not callable(transform):
                            raise RuntimeError(f"vLLM composed loader for {name!r} has no callable transform")
                        self._post_load_transforms[(id(module), name)] = transform
                    tensor.weight_loader = self._stamp(recorder, module, name, loader)

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

            if self._validate_reload:
                published_names = {tensor.name for tensor in manifest.tensors}
                consumed_names = {copy.src_name for copy in recorder.copies}
                unconsumed = sorted(published_names - consumed_names)
                coverage_mismatches: list[str] = []
                for group in groups:
                    for param_name in group.param_names:
                        parameter = getattr(group.layer, param_name)
                        coverage = torch.zeros(parameter.numel(), dtype=torch.uint8, device="cpu")
                        for copy in group.copies:
                            if copy.param_name != param_name:
                                continue
                            coverage.as_strided(copy.shape, copy.stride, copy.offset).add_(1)
                        missing = torch.count_nonzero(coverage == 0).item()
                        overlapping = torch.count_nonzero(coverage > 1).item()
                        if missing or overlapping:
                            coverage_mismatches.append(
                                f"{self._layer_names.get(id(group.layer), type(group.layer).__name__)}."
                                f"{param_name}: missing={missing}/{parameter.numel()} overlapping={overlapping}"
                            )
                logger.info(
                    "NIXL bake coverage: rank=%d published=%d consumed=%d copies=%d unconsumed=%d "
                    "destination_coverage_mismatches=%d",
                    self._rank,
                    len(published_names),
                    len(consumed_names),
                    len(recorder.copies),
                    len(unconsumed),
                    len(coverage_mismatches),
                )
                if unconsumed:
                    logger.info(
                        "NIXL rank-local bake did not consume %d published tensors; first entries: %s",
                        len(unconsumed),
                        unconsumed[:20],
                    )
                if coverage_mismatches:
                    logger.warning(
                        "NIXL destination views do not cover logical parameters exactly; first entries: %s",
                        coverage_mismatches[:20],
                    )

            self._param_layout: dict[tuple[int, str], tuple[torch.Size, torch.dtype]] = {}
            for group in groups:
                for name in group.param_names:
                    parameter = getattr(group.layer, name)
                    self._param_layout[(id(group.layer), name)] = (parameter.shape, parameter.dtype)

            for group in groups:
                self._record_kernel_graph(group, LAYERWISE_INFO.get(group.layer))

            for layer in model.modules():
                info = LAYERWISE_INFO.get(layer)
                if info is not None and info.can_load():
                    if info.kernel_tensors is not None:
                        _place_kernel_tensors(layer, info)
                    info.reset()
            if hasattr(model, "_original_do_torchao_reload"):
                model._do_torchao_reload = model._original_do_torchao_reload

        self._previous_logical_signatures: dict[int, tuple[int, int, float, float, float, float, float]] = {}
        self._previous_kernel_signatures: dict[int, tuple[int, int, float, float, float, float, float]] = {}

        if not groups:
            raise RuntimeError("vLLM consumed no published weights during the NIXL bake")
        return groups

    def _record_kernel_graph(self, group: _BakedGroup, info: Any) -> None:
        """Continue the load bake through vLLM's real kernel postprocessing."""

        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

        recorder = KernelGraphRecorder()
        for name in group.param_names:
            shape, dtype = self._param_layout[(id(group.layer), name)]
            parameter = nn.Parameter(torch.zeros(shape, dtype=dtype, device=self.device), requires_grad=False)
            setattr(group.layer, name, parameter)
            recorder.add_input(name, parameter)

        with recorder:
            for name in group.param_names:
                transform = self._post_load_transforms.get((id(group.layer), name))
                if transform is not None:
                    parameter = getattr(group.layer, name)
                    parameter.copy_(transform(parameter))

            quant_method = getattr(group.layer, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                if hasattr(group.layer, "_already_called_process_weights_after_loading"):
                    delattr(group.layer, "_already_called_process_weights_after_loading")
                quant_method.process_weights_after_loading(group.layer)

        if info is not None and info.kernel_tensors is not None:
            parameters, buffers = info.kernel_tensors
            output_names = tuple((*parameters, *buffers))
        else:
            output_names = tuple(group.param_names)
        outputs = {
            name: tensor
            for name in output_names
            if isinstance((tensor := getattr(group.layer, name, None)), torch.Tensor)
        }
        if not outputs:
            raise RuntimeError(f"vLLM kernel bake produced no tensors for {self._layer_names[id(group.layer)]}")
        group.kernel_graph = recorder.finish(outputs)
        group.kernel_outputs = {name: TensorMeta.from_tensor(tensor) for name, tensor in outputs.items()}

    def _publish_kernel_plan(self, manifest: WeightManifest, groups: list[_BakedGroup]) -> None:
        layers: list[KernelLayerPlan] = []
        for group in groups:
            if group.kernel_graph is None:
                raise RuntimeError("kernel graph was not recorded")
            inputs: list[KernelInput] = []
            for param_name in group.param_names:
                shape, dtype = self._param_layout[(id(group.layer), param_name)]
                copies = tuple(
                    KernelSourceCopy(
                        source_name=copy.src_name,
                        operations=encode_graph_value(copy.ops),
                        offset=copy.offset,
                        shape=copy.shape,
                        stride=copy.stride,
                    )
                    for copy in group.copies
                    if copy.param_name == param_name
                )
                inputs.append(
                    KernelInput(
                        name=param_name,
                        shape=tuple(shape),
                        dtype=str(dtype),
                        copies=copies,
                    )
                )
            layers.append(
                KernelLayerPlan(
                    name=self._layer_names[id(group.layer)],
                    inputs=tuple(inputs),
                    outputs=tuple(
                        KernelOutput(name=name, shape=meta.shape, dtype=str(meta.dtype))
                        for name, meta in sorted(group.kernel_outputs.items())
                    ),
                    graph=group.kernel_graph.encode(),
                )
            )

        plan = KernelPlan(
            session_id=manifest.session_id,
            epoch=manifest.epoch,
            model=manifest.model,
            rank=self._rank,
            layers=tuple(layers),
        )
        MxChannel(
            self._mx.server_url,
            manifest.session_id,
            manifest.model,
            "inference",
            "kernel_plan",
            self._rank,
        ).publish(encode_kernel_plan(plan))
        logger.info(
            "Published recorded vLLM kernel plan: rank=%d layers=%d outputs=%d",
            self._rank,
            len(layers),
            sum(len(layer.outputs) for layer in layers),
        )

    def _build_kernel_pulls(self, manifest: WeightManifest, groups: list[_BakedGroup]) -> None:
        def valid_buffers(payload: bytes) -> bool:
            buffers = decode_kernel_buffers(payload)
            return (
                buffers.session_id == manifest.session_id
                and buffers.epoch == manifest.epoch
                and buffers.model == manifest.model
                and buffers.inference_rank == self._rank
            )

        payloads = self._mx.wait_for("trainer", "kernel_buffers", 1, valid_buffers, self._timeout)
        buffers = decode_kernel_buffers(next(iter(payloads.values())))
        remote_name = self._agent.add_remote_agent(buffers.agent.metadata)
        self._agent.connect(remote_name)
        published = {tensor.name: tensor for tensor in buffers.tensors}
        local_descriptors: list[tuple[int, int, int]] = []
        remote_descriptors: list[tuple[int, int, int]] = []
        registered: set[tuple[int, int]] = set()
        self._kernel_tensors: dict[str, torch.Tensor] = {}
        self._total_pull_bytes = 0

        for group in groups:
            for output_name, meta in group.kernel_outputs.items():
                full_name = (
                    output_name
                    if self._layer_names[id(group.layer)] == "<root>"
                    else (f"{self._layer_names[id(group.layer)]}.{output_name}")
                )
                tensor = getattr(group.layer, output_name)
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(f"live vLLM kernel tensor {full_name!r} is missing")
                if tuple(tensor.shape) != meta.shape or tensor.dtype != meta.dtype:
                    raise RuntimeError(
                        f"live vLLM kernel tensor {full_name!r} is {tuple(tensor.shape)}/{tensor.dtype}; "
                        f"recorded {meta.shape}/{meta.dtype}"
                    )
                source = published.get(full_name)
                if source is None:
                    raise KeyError(f"trainer did not publish recorded kernel tensor {full_name!r}")
                if tuple(source.shape) != tuple(tensor.shape) or _torch_dtype(source.dtype) != tensor.dtype:
                    raise RuntimeError(
                        f"kernel RDMA contract mismatch for {full_name}: "
                        f"trainer={source.shape}/{source.dtype}, inference={tuple(tensor.shape)}/{tensor.dtype}"
                    )
                if len(source.segments) != 1 or source.segments[0].logical_offset != 0:
                    raise RuntimeError(f"kernel tensor {full_name!r} is not a single contiguous trainer buffer")
                segment = source.segments[0]
                num_bytes = tensor.numel() * tensor.element_size()
                if segment.numel != tensor.numel():
                    raise RuntimeError(f"kernel tensor {full_name!r} has incomplete trainer coverage")
                region = (tensor.data_ptr(), num_bytes)
                if region not in registered:
                    self._agent.register_tensor(tensor)
                    registered.add(region)
                transfers = zip_source_destination(
                    [(0, segment.address, num_bytes)],
                    tensor_runs(tensor),
                )
                for _, source_address, destination_address, length in transfers:
                    local_descriptors.append((destination_address, length, self.device.index or 0))
                    remote_descriptors.append((source_address, length, segment.device_id))
                    self._total_pull_bytes += length
                self._kernel_tensors[full_name] = tensor

        local = self._agent.prepare_local(local_descriptors)
        remote = self._agent.prepare_remote(remote_name, remote_descriptors)
        indices = list(range(len(local_descriptors)))
        self._kernel_pull_specs = [(local, remote, indices)]
        logger.info(
            "Prepared direct kernel-format NIXL pull: rank=%d tensors=%d bytes=%d",
            self._rank,
            len(self._kernel_tensors),
            self._total_pull_bytes,
        )

    @staticmethod
    def _stamp(recorder: BakeRecorder, layer: nn.Module, name: str, loader: Any):
        is_default_loader = getattr(loader, "__name__", "") == "default_weight_loader"

        def stamped(*args, **kwargs):
            recorder.current = (layer, name)
            try:
                # Several vLLM model loaders branch on identity with
                # default_weight_loader before deciding whether to pass
                # shard/expert routing arguments. The recording wrapper hides
                # that identity, so preserve its two-argument calling
                # convention explicitly.
                if is_default_loader:
                    return loader(*args[:2])
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
        self._reference_fingerprints = None
        if self._validate_reload:

            def matching_diagnostics(payload: bytes) -> bool:
                snapshot = decode_diagnostics(payload)
                return (
                    snapshot.session_id == manifest.session_id
                    and snapshot.model == manifest.model
                    and snapshot.step == step
                )

            payloads = self._mx.wait_for("trainer", "diagnostics", 1, matching_diagnostics, self._timeout)
            snapshot = decode_diagnostics(next(iter(payloads.values())))
            self._reference_fingerprints = {fingerprint.name: fingerprint for fingerprint in snapshot.tensors}
        started = time.perf_counter()
        self._pull_kernel_weights(step)
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
            f"NIXL weight update v{step}: {self._total_pull_bytes / 1e9:.2f} GB kernel bytes pulled "
            f"in {time.perf_counter() - started:.2f}s"
        )

    def _pull_kernel_weights(self, step: int) -> None:
        previous = None
        if self._validate_reload:
            previous = {name: fingerprint_tensor(name, tensor) for name, tensor in sorted(self._kernel_tensors.items())}
        handles = [
            self._agent.read(local, indices, remote, indices) for local, remote, indices in self._kernel_pull_specs
        ]
        for handle in handles:
            self._agent.wait(handle, context=f"kernel weight pull v{step}", timeout=self._timeout)
        torch.cuda.synchronize(self.device)
        if previous is not None:
            changed = 0
            unchanged: list[str] = []
            for name, tensor in sorted(self._kernel_tensors.items()):
                if fingerprint_tensor(name, tensor) != previous[name]:
                    changed += 1
                elif len(unchanged) < 20:
                    unchanged.append(name)
            logger.info(
                "NIXL direct kernel diagnostics v%d: rank=%d changed=%d/%d unchanged_sample=%s",
                step,
                self._rank,
                changed,
                len(self._kernel_tensors),
                unchanged,
            )

    def _reload_groups(self, step: int) -> None:
        from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
        from vllm.model_executor.model_loader.reload.layerwise import (
            LAYERWISE_INFO,
            _copy_and_restore_kernel_tensors,
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        model = self.raw_model
        logical_signatures: dict[int, tuple[int, int, float, float, float, float, float]] = {}
        kernel_tensor_sets: dict[int, tuple[_BakedGroup, tuple[str, ...]]] = {}
        write_checks = 0
        write_mismatch_count = 0
        write_mismatches: list[str] = []
        dtype_casts: dict[tuple[torch.dtype, torch.dtype], int] = defaultdict(int)
        source_checks = 0
        source_partial = 0
        source_value_mismatch_count = 0
        source_layout_mismatch_count = 0
        source_value_mismatches: list[str] = []
        source_layout_mismatches: list[str] = []
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(model)
            for group in self._groups:
                info = LAYERWISE_INFO.get(group.layer)
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
                    # Weight loaders are allowed to populate only the logical
                    # region of a kernel tensor.  FusedMoE, for example, pads
                    # Nemotron's intermediate dimension from 1856 to 1920 and
                    # leaves the padding at zero.  Re-materializing with
                    # ``empty`` makes that padding undefined and can feed NaNs
                    # into the kernel after an otherwise correct reload.
                    setattr(group.layer, name, nn.Parameter(torch.zeros(shape, dtype=dtype, device=self.device), False))
                for index, copy in enumerate(group.copies):
                    source_dtype = group.arena_dtypes[index]
                    num_bytes = prod(copy.shape) * source_dtype.itemsize
                    source = (
                        self._arena.narrow(0, group.arena_offsets[index], num_bytes).view(source_dtype).view(copy.shape)
                    )
                    if self._validate_reload and self._reference_fingerprints is not None:
                        reference = self._reference_fingerprints.get(copy.src_name)
                        if reference is None or reference.numel != source.numel():
                            source_partial += 1
                        else:
                            actual = fingerprint_tensor(copy.src_name, source)
                            source_checks += 1
                            invariant_matches = (
                                actual.word_sum == reference.word_sum
                                and actual.word_square_sum == reference.word_square_sum
                            )
                            if not invariant_matches:
                                source_value_mismatch_count += 1
                                if len(source_value_mismatches) < 20:
                                    source_value_mismatches.append(
                                        f"{copy.src_name}: expected=(sum={reference.word_sum},"
                                        f"sq={reference.word_square_sum}) "
                                        f"actual=(sum={actual.word_sum},sq={actual.word_square_sum})"
                                    )
                            elif actual.samples != reference.samples:
                                source_layout_mismatch_count += 1
                                if len(source_layout_mismatches) < 20:
                                    source_layout_mismatches.append(copy.src_name)
                    destination = getattr(group.layer, copy.param_name).as_strided(
                        copy.shape,
                        copy.stride,
                        copy.offset,
                    )
                    destination.copy_(source)

                # Validate the actual logical destinations after *all* writes
                # for this layer.  Checking only immediately after copy_ would
                # miss overlapping destination views, where a later loader
                # write silently clobbers an earlier one.  This boundary also
                # distinguishes correct RDMA payloads from a bad destination
                # offset/stride or an unintended dtype conversion before any
                # vLLM kernel processing runs.
                if self._validate_reload:
                    for index, copy in enumerate(group.copies):
                        source_dtype = group.arena_dtypes[index]
                        num_bytes = prod(copy.shape) * source_dtype.itemsize
                        source = (
                            self._arena.narrow(0, group.arena_offsets[index], num_bytes)
                            .view(source_dtype)
                            .view(copy.shape)
                        )
                        destination = getattr(group.layer, copy.param_name).as_strided(
                            copy.shape,
                            copy.stride,
                            copy.offset,
                        )
                        if source.dtype != destination.dtype:
                            dtype_casts[(source.dtype, destination.dtype)] += 1
                            expected = source.to(destination.dtype)
                        else:
                            expected = source
                        write_checks += 1
                        if torch.equal(destination, expected):
                            continue
                        write_mismatch_count += 1
                        changed = torch.count_nonzero(destination != expected).item()
                        nonfinite = (
                            torch.count_nonzero(~torch.isfinite(destination)).item()
                            if destination.is_floating_point()
                            else 0
                        )
                        max_abs = (
                            (destination.float() - expected.float()).abs().max().item()
                            if destination.is_floating_point() and destination.numel()
                            else None
                        )
                        if len(write_mismatches) < 20:
                            write_mismatches.append(
                                f"{self._layer_names.get(id(group.layer), type(group.layer).__name__)}."
                                f"{copy.param_name} <- {copy.src_name}: shape={copy.shape} "
                                f"stride={copy.stride} offset={copy.offset} "
                                f"src_dtype={source.dtype} dst_dtype={destination.dtype} "
                                f"changed={changed}/{destination.numel()} nonfinite={nonfinite} max_abs={max_abs}"
                            )

                # vLLM's composed loaders first load a tensor and then apply
                # a destination-side transform.  The lazy bake observes the
                # source copy but cannot see operations on the materialized
                # destination (for example Nemotron's A = -exp(A_log)).
                # Replay that recorded transform over the already-sharded
                # destination so TP/EP ranks do not need the full HF tensor.
                for name in group.param_names:
                    transform = self._post_load_transforms.get((id(group.layer), name))
                    if transform is not None:
                        parameter = getattr(group.layer, name)
                        parameter.copy_(transform(parameter))

                if self._validate_reload:
                    logical_signatures[id(group.layer)] = _tensor_set_signature(group.layer, group.param_names)

                quant_method = getattr(group.layer, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    if hasattr(group.layer, "_already_called_process_weights_after_loading"):
                        delattr(group.layer, "_already_called_process_weights_after_loading")
                    quant_method.process_weights_after_loading(group.layer)
                expected = None
                if self._validate_reload and info is not None and info.kernel_tensors is not None:
                    parameters, buffers = info.kernel_tensors
                    expected = {}
                    for name in (*parameters, *buffers):
                        tensor = getattr(group.layer, name, None)
                        if isinstance(tensor, torch.Tensor):
                            expected[name] = tensor.detach().clone()
                if info is not None and info.kernel_tensors is not None:
                    _copy_and_restore_kernel_tensors(group.layer, info)
                if expected is not None:
                    for name, expected_tensor in expected.items():
                        actual = getattr(group.layer, name, None)
                        if not isinstance(actual, torch.Tensor) or torch.equal(actual, expected_tensor):
                            continue
                        changed = torch.count_nonzero(actual != expected_tensor).item()
                        nonfinite = (
                            torch.count_nonzero(~torch.isfinite(actual)).item() if actual.is_floating_point() else 0
                        )
                        max_abs = (
                            (actual.float() - expected_tensor.float()).abs().max().item()
                            if actual.is_floating_point() and actual.numel()
                            else None
                        )
                        logger.warning(
                            "NIXL reload validation mismatch: layer=%s tensor=%s "
                            "shape=%s changed=%d/%d nonfinite=%d max_abs=%s",
                            self._layer_names.get(id(group.layer), type(group.layer).__name__),
                            name,
                            tuple(actual.shape),
                            changed,
                            actual.numel(),
                            nonfinite,
                            max_abs,
                        )
                if self._validate_reload:
                    if info is not None and info.kernel_tensors is not None:
                        parameters, buffers = info.kernel_tensors
                        kernel_names = tuple((*parameters, *buffers))
                    else:
                        kernel_names = tuple(group.param_names)
                    kernel_tensor_sets[id(group.layer)] = (group, kernel_names)
                if info is not None:
                    info.reset()
            finalize_layerwise_reload(model, self.model_runner.model_config)

        if self._validate_reload:
            logical_changed = 0
            kernel_changed = 0
            logical_changed_kernel_unchanged: list[str] = []
            logical_unchanged_kernel_changed: list[str] = []
            for layer_id, logical_signature in logical_signatures.items():
                group, kernel_names = kernel_tensor_sets[layer_id]
                kernel_signature = _tensor_set_signature(group.layer, kernel_names)
                previous_logical = self._previous_logical_signatures.get(layer_id)
                previous_kernel = self._previous_kernel_signatures.get(layer_id)
                did_logical_change = previous_logical is not None and logical_signature != previous_logical
                did_kernel_change = previous_kernel is not None and kernel_signature != previous_kernel
                logical_changed += int(did_logical_change)
                kernel_changed += int(did_kernel_change)
                layer_name = self._layer_names.get(layer_id, type(group.layer).__name__)
                if did_logical_change and not did_kernel_change:
                    logical_changed_kernel_unchanged.append(layer_name)
                elif did_kernel_change and not did_logical_change:
                    logical_unchanged_kernel_changed.append(layer_name)
                self._previous_logical_signatures[layer_id] = logical_signature
                self._previous_kernel_signatures[layer_id] = kernel_signature

            logger.info(
                "NIXL reload diagnostics v%d: rank=%d groups=%d logical_changed=%d "
                "kernel_changed=%d logical_changed_kernel_unchanged=%d "
                "logical_unchanged_kernel_changed=%d write_checks=%d write_mismatches=%d dtype_casts=%s",
                step,
                self._rank,
                len(logical_signatures),
                logical_changed,
                kernel_changed,
                len(logical_changed_kernel_unchanged),
                len(logical_unchanged_kernel_changed),
                write_checks,
                write_mismatch_count,
                {f"{source}->{destination}": count for (source, destination), count in dtype_casts.items()},
            )
            if write_mismatches:
                logger.warning("NIXL logical destination write mismatches; first copies: %s", write_mismatches)
            logger.info(
                "NIXL source reference diagnostics v%d: rank=%d checked=%d partial=%d value_mismatches=%d "
                "layout_mismatches=%d",
                step,
                self._rank,
                source_checks,
                source_partial,
                source_value_mismatch_count,
                source_layout_mismatch_count,
            )
            if source_value_mismatches:
                logger.warning("NIXL pulled source value mismatches; first tensors: %s", source_value_mismatches)
            if source_layout_mismatches:
                logger.warning("NIXL pulled source layout mismatches; first tensors: %s", source_layout_mismatches)
            if logical_changed_kernel_unchanged:
                logger.warning(
                    "NIXL logical inputs changed but kernel tensors did not; first layers: %s",
                    logical_changed_kernel_unchanged[:20],
                )
            if logical_unchanged_kernel_changed:
                logger.warning(
                    "NIXL kernel tensors changed without changed logical inputs; first layers: %s",
                    logical_unchanged_kernel_changed[:20],
                )
