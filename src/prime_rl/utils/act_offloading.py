from contextlib import nullcontext

import psutil
import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import StorageWeakRef

from prime_rl.configs.trainer import ActivationOffloadingConfig

from .logger import get_logger


class OffloadActivations(saved_tensors_hooks):
    """Offload tensors saved for backward to pinned CPU memory, copying on a side stream.

    In the forward pass, at most ``max_inflight`` activations stay on the GPU while their copies
    to CPU run. In the backward pass, the next ``max_inflight`` activations are copied back
    ahead of use. GPU memory is always allocated on the compute stream, and the compute stream
    waits on copy events before it reads that memory.

    Tensors that share storage with a parameter of ``model`` stay on the GPU. A compiled block saves
    the local tensors of unsharded DTensor parameters (e.g. EP experts), and FSDP frees and refills
    that storage itself.

    A contiguous tensor saved again (same storage, offset, numel and dtype, e.g. RoPE ``cos``/``sin``
    or shared indexer outputs saved by every block) reuses the first save's copy in both directions.
    The key leaves out the version counter: compiled blocks can save views of one storage with
    different versions, and a mutation between two saves already fails autograd's version check.
    """

    def __init__(self, model: nn.Module, max_inflight: int = 5, min_offload_size: int = 1024) -> None:
        self.param_slots = [(module, name) for module in model.modules() for name in module._parameters]
        self.max_inflight = max_inflight
        self.min_offload_size = min_offload_size
        self.compute_stream = torch.cuda.current_stream()
        self.copy_stream = torch.cuda.Stream()
        self.next_id = 0
        self.kept: dict[int, torch.Tensor] = {}
        self.on_cpu: dict[int, torch.Tensor] = {}
        self.offloading: dict[int, tuple[torch.Tensor, torch.Event]] = {}
        self.prefetched: dict[int, tuple[torch.Tensor, torch.Event]] = {}
        self.copies: dict[tuple, tuple[StorageWeakRef, int]] = {}  # storage key -> (storage ref, copy id)
        self.aliases: dict[int, tuple[int, torch.Size]] = {}  # tensor id -> (copy id, shape) for repeated saves
        self.refs: dict[int, int] = {}  # copy id -> saves not yet unpacked
        super().__init__(self.pack, self.unpack)

    def __exit__(self, *args: object) -> None:
        super().__exit__(*args)
        ram_pct = psutil.virtual_memory().percent
        if ram_pct > 90:
            get_logger().warning(f"Host RAM is {ram_pct}% used after offloading activations")

    def pack(self, tensor: torch.Tensor) -> int:
        self.next_id += 1
        tensor_id = self.next_id
        if (
            tensor.device.type != "cuda"
            or tensor.nbytes < self.min_offload_size
            or isinstance(tensor, (nn.Parameter, nn.Buffer))
            or tensor.untyped_storage()._cdata in self.param_storages()
        ):
            self.kept[tensor_id] = tensor
            return tensor_id

        if tensor.is_contiguous():
            key = (tensor.untyped_storage()._cdata, tensor.storage_offset(), tensor.numel(), tensor.dtype)
            match = self.copies.get(key)
            if match is not None and not match[0].expired() and match[1] in self.refs:
                self.refs[match[1]] += 1
                self.aliases[tensor_id] = (match[1], tensor.shape)
                return tensor_id
            self.copies[key] = (StorageWeakRef(tensor.untyped_storage()), tensor_id)

        while len(self.offloading) >= self.max_inflight:
            _, event = self.offloading.pop(next(iter(self.offloading)))
            self.compute_stream.wait_event(event)

        self.copy_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.copy_stream):
            cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu_tensor.copy_(tensor, non_blocking=True)
        tensor.record_stream(self.copy_stream)
        self.on_cpu[tensor_id] = cpu_tensor
        self.offloading[tensor_id] = (tensor, self.copy_stream.record_event())
        self.refs[tensor_id] = 1
        return tensor_id

    def param_storages(self) -> set[int]:
        storages = set()
        with torch.no_grad():
            for module, name in self.param_slots:
                param = module._parameters[name]
                if param is not None:
                    local = param.to_local() if isinstance(param, DTensor) else param
                    storages.add(local.untyped_storage()._cdata)
        return storages

    def unpack(self, tensor_id: int) -> torch.Tensor:
        if tensor_id in self.kept:
            return self.kept.pop(tensor_id)

        copy_id, shape = self.aliases.pop(tensor_id, (tensor_id, None))
        if copy_id in self.offloading:
            tensor, event = self.offloading[copy_id]
        else:
            if copy_id not in self.prefetched:
                self.prefetch(copy_id)
            tensor, event = self.prefetched[copy_id]
        self.refs[copy_id] -= 1
        if self.refs[copy_id] == 0:
            del self.refs[copy_id], self.on_cpu[copy_id]
            self.offloading.pop(copy_id, None)
            self.prefetched.pop(copy_id, None)

        upcoming = [i for i in reversed(self.on_cpu) if i < tensor_id and i not in self.offloading]
        for i in upcoming[: self.max_inflight]:
            if i not in self.prefetched:
                self.prefetch(i)

        self.compute_stream.wait_event(event)
        return tensor if shape is None else tensor.view(shape)

    def prefetch(self, tensor_id: int) -> None:
        cpu_tensor = self.on_cpu[tensor_id]
        tensor = torch.empty_like(cpu_tensor, device="cuda")
        self.copy_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.copy_stream):
            tensor.copy_(cpu_tensor, non_blocking=True)
        tensor.record_stream(self.copy_stream)
        self.prefetched[tensor_id] = (tensor, self.copy_stream.record_event())


def maybe_activation_offloading(
    config: ActivationOffloadingConfig | None, model: nn.Module
) -> OffloadActivations | nullcontext:
    if config is None:
        return nullcontext()

    return OffloadActivations(model, max_inflight=config.max_inflight_activations)
