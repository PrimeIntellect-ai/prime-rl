"""Activation offloading for the asynchronous pipeline schedules.

A pipeline stage keeps the activations of every micro-batch between its forward and its backward: the
stage's received inputs and each block's checkpointed inputs. Most of that time a micro-batch's
activations sit idle, so after its forward the offloader copies their storages to pinned host memory
and frees them on the GPU, and a few ops before its backward copies them back into the same storages.
Tensors keep their identity (only their storage is emptied and refilled, as FSDP does with parameters),
so autograd's saved tensors, the stage's cached inputs and its receive buffers need no changes.

Offloaded are the tensors passed into the stage and into its decoder layers and engram layers (what
activation checkpointing keeps), not the stage's outputs, which are still being sent, nor parameters.
"""

import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_flatten

from prime_rl.utils.vlm import get_language_model

Key = tuple[int, int]  # (stage, micro-batch)


class PipelineActivationOffloader:
    """Moves the activations of micro-batches `(stage, mb)` to host memory between their forward and backward.

    The schedule calls `forward(key)` around a forward to collect the micro-batch's activations,
    `swap_out(key, output)` right after it, `prefetch(key)` a few ops before its backward and
    `wait(key)` right before it. Copies run on two side streams; the compute stream only waits for a
    micro-batch's activations to be back before its backward."""

    def __init__(self, model_parts: list[nn.Module], min_bytes: int):
        self.min_bytes = min_bytes
        self._d2h, self._h2d = torch.cuda.Stream(), torch.cuda.Stream()
        self._collecting: Key | None = None
        self._candidates: dict[Key, dict[int, Tensor]] = {}
        # Per offloaded micro-batch: (storage, pinned host copy) pairs, and the event of their return
        self._host: dict[Key, list[tuple[torch.UntypedStorage, Tensor]]] = {}
        self._returned: dict[Key, tuple[torch.cuda.Event, list[Tensor]]] = {}
        for part in model_parts:
            part.register_forward_pre_hook(self._collect, with_kwargs=True)
            language_model = get_language_model(part)
            for name in ("layers", "engrams"):
                for module in getattr(language_model, name, nn.ModuleDict()).children():
                    module.register_forward_pre_hook(self._collect, with_kwargs=True)

    def _collect(self, _module: nn.Module, args: tuple, kwargs: dict) -> None:
        # Recomputed forwards (activation checkpointing in backward) run outside `forward` and are skipped.
        if self._collecting is None:
            return
        candidates = self._candidates.setdefault(self._collecting, {})
        for tensor in tree_flatten((args, kwargs))[0]:
            if (
                isinstance(tensor, Tensor)
                and tensor.is_cuda
                and not isinstance(tensor, nn.Parameter)
                and tensor.untyped_storage().nbytes() >= self.min_bytes
            ):
                candidates.setdefault(tensor.untyped_storage().data_ptr(), tensor)

    def forward(self, key: Key) -> "_Collecting":
        return _Collecting(self, key)

    def swap_out(self, key: Key, output) -> None:
        """Copy the micro-batch's collected activations to host memory and free them on the GPU."""
        candidates = self._candidates.pop(key, {})
        for tensor in tree_flatten(output)[0]:
            if isinstance(tensor, Tensor):
                candidates.pop(tensor.untyped_storage().data_ptr(), None)
        if not candidates:
            return
        self._d2h.wait_stream(torch.cuda.current_stream())
        pairs = []
        with torch.cuda.stream(self._d2h):
            for tensor in candidates.values():
                storage = tensor.untyped_storage()
                flat = _bytes_view(storage, tensor.device)
                host = torch.empty(storage.nbytes(), dtype=torch.uint8, pin_memory=True)
                host.copy_(flat, non_blocking=True)
                # The allocator reuses the block only after the copy is done.
                flat.record_stream(self._d2h)
                storage.resize_(0)
                pairs.append((storage, host))
        self._host[key] = pairs

    def prefetch(self, key: Key) -> None:
        """Start copying the micro-batch's activations back into their storages."""
        pairs = self._host.pop(key, None)
        if pairs is None:
            return
        device = torch.device("cuda", torch.cuda.current_device())
        flats = []
        # Allocated from the copy stream's own pool, which only ever holds these few same-sized buffers, so the
        # compute stream's pool does not fragment around them.
        with torch.cuda.stream(self._h2d):
            for storage, host in pairs:
                storage.resize_(host.numel())
                flat = _bytes_view(storage, device)
                flat.copy_(host, non_blocking=True)
                flats.append(flat)
        self._returned[key] = (self._h2d.record_event(), flats)

    def wait(self, key: Key) -> None:
        """Make the compute stream wait until the micro-batch's activations are back."""
        if key in self._host:
            self.prefetch(key)
        returned = self._returned.pop(key, None)
        if returned is not None:
            event, flats = returned
            torch.cuda.current_stream().wait_event(event)
            # Freed after the backward that reads them: the copy stream's pool reuses them only after it.
            for flat in flats:
                flat.record_stream(torch.cuda.current_stream())


class _Collecting:
    def __init__(self, offloader: PipelineActivationOffloader, key: Key):
        self.offloader, self.key = offloader, key

    def __enter__(self) -> None:
        self.offloader._collecting = self.key

    def __exit__(self, *exc) -> None:
        self.offloader._collecting = None


def _bytes_view(storage: torch.UntypedStorage, device: torch.device) -> Tensor:
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage)


def offload_plan(run_ops: list[tuple[str, int, int]], stages: set[int] | None, prefetch_ahead: int):
    """For a rank's run ops in order: the forwards whose activations go to host memory (their backward is
    more than `prefetch_ahead` ops later, on an offloading stage), and for each op position the backwards
    to prefetch before it."""
    position = {op: i for i, op in enumerate(run_ops)}
    offloaded, prefetch_at = set(), {}
    for (kind, stage, mb), i in position.items():
        if kind != "F" or (stages is not None and stage not in stages):
            continue
        backward = position.get(("B", stage, mb))
        if backward is not None and backward - i > prefetch_ahead:
            offloaded.add((stage, mb))
            prefetch_at.setdefault(backward - prefetch_ahead, []).append((stage, mb))
    return offloaded, prefetch_at
