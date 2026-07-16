"""Trainer-side executor for inference-recorded vLLM kernel plans."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import prod
from typing import Any

import torch

from prime_rl.weight_transfer.chains import region_elem_runs, resolve_chain_region, tensor_runs
from prime_rl.weight_transfer.kernel_graph import KernelGraph, decode_graph_value
from prime_rl.weight_transfer.nixl import NixlAgent
from prime_rl.weight_transfer.publication import route_published_region
from prime_rl.weight_transfer.sharding import zip_source_destination
from prime_rl.weight_transfer.wire import (
    AgentDescriptor,
    KernelBufferManifest,
    KernelLayerPlan,
    KernelPlan,
    PublishedTensor,
    TensorSegment,
    WeightManifest,
)


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {name!r}")
    return value


def _align(offset: int, alignment: int = 256) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _full_name(layer: str, tensor: str) -> str:
    return tensor if layer == "<root>" else f"{layer}.{tensor}"


@dataclass
class _CopyExecution:
    input_name: str
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    staging: torch.Tensor


@dataclass
class _LayerExecution:
    plan: KernelLayerPlan
    graph: KernelGraph
    inputs: dict[str, torch.Tensor]
    copies: list[_CopyExecution]
    outputs: dict[str, torch.Tensor]
    local_copies: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    pulls: list[tuple[Any, Any, list[int]]] = field(default_factory=list)


class KernelPlanExecutor:
    """Materialize exact kernel-format buffers for one inference rank.

    Source HF slices are pulled from the already-registered sharded Prime
    buffers.  Loader copies (including dtype conversion) populate logical
    vLLM inputs, then the inference-recorded post-load graph produces the
    final tensors that inference pulls directly into live storage.
    """

    def __init__(
        self,
        *,
        plan: KernelPlan,
        source_manifest: WeightManifest,
        agent: NixlAgent,
        local_source_buffers: tuple[torch.Tensor, ...],
        device: torch.device,
        timeout: float,
    ) -> None:
        self.plan = plan
        self.source_manifest = source_manifest
        self.agent = agent
        self.device = device
        self.timeout = timeout
        self._sources = {tensor.name: tensor for tensor in source_manifest.tensors}
        self._local_source_buffers = local_source_buffers

        local_agent_indices = [
            index for index, descriptor in enumerate(source_manifest.agents) if descriptor.name == agent.name
        ]
        if len(local_agent_indices) != 1:
            raise RuntimeError(
                f"trainer agent {agent.name!r} appears {len(local_agent_indices)} times in source manifest"
            )
        self._local_agent = local_agent_indices[0]

        arena_bytes = max((self._layer_arena_bytes(layer) for layer in plan.layers), default=1)
        self._arena = torch.empty(arena_bytes, dtype=torch.uint8, device=device)
        self.agent.register_tensor(self._arena)
        self.layers = [self._build_layer(layer) for layer in plan.layers]

    def _layer_arena_bytes(self, layer: KernelLayerPlan) -> int:
        offset = 0
        for input_ in layer.inputs:
            offset = _align(offset)
            offset += prod(input_.shape) * _dtype(input_.dtype).itemsize
        for input_ in layer.inputs:
            for copy in input_.copies:
                source = self._sources[copy.source_name]
                offset = _align(offset)
                offset += prod(copy.shape) * _dtype(source.dtype).itemsize
        return offset

    def _view(self, offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        num_bytes = prod(shape) * dtype.itemsize
        return self._arena.narrow(0, offset, num_bytes).view(dtype).view(shape)

    def _build_layer(self, plan: KernelLayerPlan) -> _LayerExecution:
        offset = 0
        inputs: dict[str, torch.Tensor] = {}
        for input_ in plan.inputs:
            offset = _align(offset)
            dtype = _dtype(input_.dtype)
            inputs[input_.name] = self._view(offset, input_.shape, dtype)
            offset += prod(input_.shape) * dtype.itemsize

        copies: list[_CopyExecution] = []
        copy_sources: list[tuple[Any, torch.Tensor]] = []
        for input_ in plan.inputs:
            for copy in input_.copies:
                source = self._sources.get(copy.source_name)
                if source is None:
                    raise KeyError(f"kernel plan requested unpublished HF tensor {copy.source_name!r}")
                source_dtype = _dtype(source.dtype)
                offset = _align(offset)
                staging = self._view(offset, copy.shape, source_dtype)
                offset += prod(copy.shape) * source_dtype.itemsize
                copies.append(_CopyExecution(input_.name, copy.offset, copy.shape, copy.stride, staging))
                copy_sources.append((copy, staging))

        graph = KernelGraph.decode(plan.graph)
        if set(graph.input_names) != set(inputs):
            raise RuntimeError(
                f"recorded graph/input mismatch for {plan.name}: graph={graph.input_names}, plan={tuple(inputs)}"
            )

        outputs: dict[str, torch.Tensor] = {}
        for output in plan.outputs:
            full_name = _full_name(plan.name, output.name)
            tensor = torch.empty(output.shape, dtype=_dtype(output.dtype), device=self.device)
            self.agent.register_tensor(tensor)
            outputs[full_name] = tensor

        execution = _LayerExecution(plan=plan, graph=graph, inputs=inputs, copies=copies, outputs=outputs)
        self._build_source_pulls(execution, copy_sources)
        return execution

    def _build_source_pulls(self, execution: _LayerExecution, copies: list[tuple[Any, torch.Tensor]]) -> None:
        local_descriptors: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        remote_descriptors: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

        for copy, staging in copies:
            source = self._sources[copy.source_name]
            source_dtype = _dtype(source.dtype)
            source_offset, shape, stride = resolve_chain_region(
                source.shape,
                source_dtype,
                decode_graph_value(copy.operations),
            )
            pieces = route_published_region(
                source,
                region_elem_runs(source_offset, shape, stride),
                source_dtype.itemsize,
            )
            transfers = zip_source_destination(pieces, tensor_runs(staging))
            for owner, source_address, destination_address, num_bytes in transfers:
                if owner == self._local_agent:
                    source_view = self._local_bytes(source_address, num_bytes)
                    destination_offset = destination_address - self._arena.data_ptr()
                    destination_view = self._arena.narrow(0, destination_offset, num_bytes)
                    execution.local_copies.append((source_view, destination_view))
                else:
                    local_descriptors[owner].append((destination_address, num_bytes, self.device.index or 0))
                    remote_device = self._remote_device(owner, source_address)
                    remote_descriptors[owner].append((source_address, num_bytes, remote_device))

        for owner, descriptors in sorted(remote_descriptors.items()):
            remote_name = self.agent.add_remote_agent(self.source_manifest.agents[owner].metadata)
            self.agent.connect(remote_name)
            local = self.agent.prepare_local(local_descriptors[owner])
            remote = self.agent.prepare_remote(remote_name, descriptors)
            execution.pulls.append((local, remote, list(range(len(descriptors)))))

    def _local_bytes(self, address: int, num_bytes: int) -> torch.Tensor:
        for tensor in self._local_source_buffers:
            lower = tensor.data_ptr()
            upper = lower + tensor.numel() * tensor.element_size()
            if lower <= address and address + num_bytes <= upper:
                return tensor.view(torch.uint8).reshape(-1).narrow(0, address - lower, num_bytes)
        raise RuntimeError(f"local source address [{address}, {address + num_bytes}) is not in a registered buffer")

    def _remote_device(self, owner: int, address: int) -> int:
        for tensor in self.source_manifest.tensors:
            for segment in tensor.segments:
                lower = segment.address
                upper = lower + segment.numel * _dtype(tensor.dtype).itemsize
                if segment.agent == owner and lower <= address < upper:
                    return segment.device_id
        raise RuntimeError(f"no remote device descriptor for trainer agent {owner} address {address}")

    @torch.no_grad()
    def materialize(self) -> None:
        for layer in self.layers:
            for tensor in layer.inputs.values():
                tensor.zero_()
            for source, destination in layer.local_copies:
                destination.copy_(source, non_blocking=True)
            handles = [self.agent.read(local, indices, remote, indices) for local, remote, indices in layer.pulls]
            for handle in handles:
                self.agent.wait(handle, context=f"kernel input pull for {layer.plan.name}", timeout=self.timeout)

            for copy in layer.copies:
                destination = layer.inputs[copy.input_name].as_strided(copy.shape, copy.stride, copy.offset)
                destination.copy_(copy.staging)

            produced = layer.graph.replay(layer.inputs)
            for output_name, tensor in produced.items():
                full_name = _full_name(layer.plan.name, output_name)
                destination = layer.outputs[full_name]
                if tuple(tensor.shape) != tuple(destination.shape) or tensor.dtype != destination.dtype:
                    raise RuntimeError(
                        f"kernel output {full_name} has {tuple(tensor.shape)}/{tensor.dtype}; "
                        f"destination is {tuple(destination.shape)}/{destination.dtype}"
                    )
                destination.copy_(tensor)
        torch.cuda.synchronize(self.device)

    def buffer_manifest(self) -> KernelBufferManifest:
        tensors: list[PublishedTensor] = []
        for layer in self.layers:
            for name, tensor in layer.outputs.items():
                tensors.append(
                    PublishedTensor(
                        name=name,
                        dtype=str(tensor.dtype),
                        shape=tuple(tensor.shape),
                        segments=(
                            TensorSegment(
                                agent=0,
                                logical_offset=0,
                                numel=tensor.numel(),
                                address=tensor.data_ptr(),
                                device_id=tensor.device.index or 0,
                            ),
                        ),
                    )
                )
        return KernelBufferManifest(
            session_id=self.plan.session_id,
            epoch=self.plan.epoch,
            model=self.plan.model,
            inference_rank=self.plan.rank,
            agent=AgentDescriptor(self.agent.name, self.agent.metadata()),
            tensors=tuple(sorted(tensors, key=lambda tensor: tensor.name)),
        )
