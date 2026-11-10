import copy
import math

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from prime_rl.trainer.optim.base import OffloadOptimizer
from prime_rl.trainer.optim.subset_step import SubsetStep, subset_step_for
from prime_rl.utils.logger import get_logger

# Byte budget of one staging chunk. A single state larger than this gets a chunk of its own.
CHUNK_BYTES = 1 << 30
# Alignment of each state view inside a staging buffer.
ALIGN_BYTES = 512
# An overlapped step enqueues this many chunks up front, so the copies run while the trainer
# finishes its step bookkeeping, and later stays this many chunks ahead of the forward. Enqueuing
# every chunk at once would fill the CUDA launch queue and block the CPU until the copies drain.
INITIAL_CHUNKS = 8
LOOKAHEAD_CHUNKS = 4


def _local(value: torch.Tensor) -> torch.Tensor:
    return value._local_tensor if isinstance(value, DTensor) else value


def _with_local(value: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, DTensor):
        return local
    new_dtensor = copy.copy(value)
    new_dtensor._local_tensor = local
    return new_dtensor


def _nominal_bytes(value: torch.Tensor) -> int:
    """Size of a tensor's local shard, the same on every rank: an even split of each sharded mesh dim."""
    numel = value.numel()
    if isinstance(value, DTensor):
        shards = math.prod(
            value.device_mesh.size(dim) for dim, placement in enumerate(value.placements) if placement.is_shard()
        )
        numel = -(-numel // shards)
    return numel * value.element_size()


class _ChunkedStep:
    """One chunked optimizer step. Chunks stream through two GPU staging buffers: while chunk i updates,
    chunk i+1 loads on the h2d stream and chunk i-1 writes back to pinned memory on the d2h stream.

    Chunks are enqueued on demand by ``advance``. ``subset_step`` captured the step's hyperparameters
    and gradients up front, so a chunk enqueued after the scheduler step or ``zero_grad`` updates
    exactly as it would have at step time."""

    def __init__(
        self,
        optimizer: Optimizer,
        subset_step: SubsetStep,
        chunks: list[tuple[list[torch.Tensor], list[tuple[torch.Tensor, str, int]]]],
        buffer_bytes: int,
        h2d: torch.cuda.Stream,
        d2h: torch.cuda.Stream,
        update: torch.cuda.Stream,
    ):
        self.optimizer = optimizer
        self.subset_step = subset_step
        self.chunks = chunks
        self.h2d, self.d2h, self.update = h2d, d2h, update
        compute = torch.cuda.current_stream()
        self.overlapped = update != compute
        # Allocated on the compute stream, so the caching allocator reuses the same blocks every step.
        self.buffers = [torch.empty(buffer_bytes, dtype=torch.uint8, device="cuda") for _ in range(2)]
        h2d.wait_stream(compute)
        # The previous step's writebacks must land before the loads read the CPU states again.
        h2d.wait_stream(d2h)
        update.wait_stream(compute)
        self.grads = {param: param.grad for params, _ in chunks for param in params}
        self.written_back: list[torch.cuda.Event | None] = [None, None]
        self.cpu_values: list[dict[tuple[torch.Tensor, str], torch.Tensor]] = [{}, {}]
        self.updated: list[torch.cuda.Event] = []
        self.loaded = self._load(0)

    @property
    def done(self) -> bool:
        return len(self.updated) == len(self.chunks)

    def _load(self, i: int) -> torch.cuda.Event:
        state = self.optimizer.state
        buffer, slot = self.buffers[i % 2], self.cpu_values[i % 2]
        with torch.cuda.stream(self.h2d):
            if self.written_back[i % 2] is not None:
                self.h2d.wait_event(self.written_back[i % 2])
            slot.clear()
            for param, key, offset in self.chunks[i][1]:
                cpu_value = state[param][key]
                cpu_local = _local(cpu_value)
                nbytes = cpu_local.numel() * cpu_local.element_size()
                gpu_local = buffer[offset : offset + nbytes].view(cpu_local.dtype).view(cpu_local.shape)
                gpu_local.copy_(cpu_local, non_blocking=True)
                slot[param, key] = cpu_value
                state[param][key] = _with_local(cpu_value, gpu_local)
            return self.h2d.record_event()

    def advance(self, until: int) -> None:
        """Enqueue every chunk up to and including chunk ``until``."""
        state = self.optimizer.state
        while not self.done and len(self.updated) <= until:
            i = len(self.updated)
            params = self.chunks[i][0]
            next_loaded = self._load(i + 1) if i + 1 < len(self.chunks) else None
            self.update.wait_event(self.loaded)
            with torch.cuda.stream(self.update):
                self.subset_step.step_params(params)
                self.updated.append(self.update.record_event())
            self.d2h.wait_stream(self.update)
            with torch.cuda.stream(self.d2h):
                for (param, key), cpu_value in self.cpu_values[i % 2].items():
                    _local(cpu_value).copy_(_local(state[param][key]), non_blocking=True)
                    state[param][key] = cpu_value
                self.written_back[i % 2] = self.d2h.record_event()
            self.loaded = next_loaded
            for param in params:
                grad = self.grads.pop(param)
                if self.overlapped and grad is not None:
                    # zero_grad may already have let go of it: keep the compute stream off its
                    # memory until the update has read it.
                    _local(grad).record_stream(self.update)
        if self.done and self.overlapped and self.buffers:
            for buffer in self.buffers:
                for stream in (self.h2d, self.update, self.d2h):
                    buffer.record_stream(stream)
            self.buffers = []


class CPUOffloadOptimizer(OffloadOptimizer):
    """Wraps an optimizer to keep states on CPU, moving to GPU only for step().

    Unlike FSDP's CPUOffload which offloads weights too, this keeps weights on GPU.
    With activation checkpointing, activations and optimizer states are never on GPU
    at the same time: peak memory becomes max(activations, opt_states) instead of sum.
    """

    def __init__(self, optimizer: Optimizer, overlap_model: nn.Module | None = None):
        """The step is chunked for optimizers that can update a subset of their parameters at a time
        (``subset_step.SupportsSubsetStep``, e.g. AdamW and Muon); others move all states at once.

        With ``overlap_model`` (FSDP-sharded), a chunked step returns before its chunks
        run: they update on a side stream alongside the next forward, enqueued as the forward reaches
        them, and each FSDP parameter group waits only for the chunks holding its parameters before
        its all-gather. Anything else that reads the parameters must call ``wait()`` first."""
        self.optimizer = optimizer
        self._subset_step = subset_step_for(optimizer)
        self._initialized = False
        self._cpu_buffers: dict[tuple[torch.Tensor, str], torch.Tensor] = {}
        self._streams: tuple[torch.cuda.Stream, torch.cuda.Stream, torch.cuda.Stream] | None = None
        self._fsdp_groups: dict[torch.Tensor, FSDPParamGroup] = {}
        # Forward order of the FSDP groups.
        self._group_order: dict[FSDPParamGroup, int] = {}
        self._in_flight: _ChunkedStep | None = None
        # Last chunk of the in-flight step holding each FSDP group's parameters.
        self._group_chunks: dict[FSDPParamGroup, int] = {}
        if overlap_model is not None and self._subset_step is not None:
            self._install_unshard_waits(overlap_model)

    def _install_unshard_waits(self, model: nn.Module) -> None:
        fsdp_groups, group_order = {}, {}
        for module in model.modules():
            if isinstance(module, FSDPModule):
                # A module holds one group per mesh (e.g. experts on the EP mesh next to dense weights).
                for group in module._get_fsdp_state()._fsdp_param_groups:
                    group_order.setdefault(group, len(group_order))
                    for fsdp_param in group.fsdp_params:
                        fsdp_groups[fsdp_param.sharded_param] = group
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        missing = sum(p not in fsdp_groups for p in params)
        if missing:
            get_logger().warning(
                f"{missing} of {len(params)} optimizer parameters are not managed by FSDP; "
                "the offloaded optimizer step will not overlap the next forward"
            )
            return
        self._fsdp_groups, self._group_order = fsdp_groups, group_order
        for group in set(fsdp_groups.values()):
            group.unshard = self._unshard_after_step(group, group.unshard)
            group.wait_for_unshard = self._wait_for_unshard_after_step(group, group.wait_for_unshard)

    def _updated(self, chunk: int) -> torch.cuda.Event:
        self._in_flight.advance(chunk + LOOKAHEAD_CHUNKS)
        return self._in_flight.updated[chunk]

    def _unshard_after_step(self, group: FSDPParamGroup, unshard):
        def wrapper(async_op: bool = False):
            chunk = self._group_chunks.get(group)
            if chunk is not None:
                event = self._updated(chunk)
                # The all-gather copy-in reads the sharded parameters on these streams.
                for stream in set(group.comm_ctx.get_all_gather_streams(async_op, group._training_state)):
                    stream.wait_event(event)
            return unshard(async_op)

        return wrapper

    def _wait_for_unshard_after_step(self, group: FSDPParamGroup, wait_for_unshard):
        def wrapper():
            chunk = self._group_chunks.pop(group, None)
            if chunk is not None:
                torch.cuda.current_stream().wait_event(self._updated(chunk))
            return wait_for_unshard()

        return wrapper

    def _finish_in_flight(self) -> None:
        if self._in_flight is not None:
            self._in_flight.advance(len(self._in_flight.chunks))
            self._in_flight = None
            self._group_chunks.clear()

    def wait(self) -> None:
        """Make the current stream wait for an overlapped step's updates and state writebacks."""
        self._finish_in_flight()
        if self._streams is not None:
            current = torch.cuda.current_stream()
            for stream in self._streams:
                current.wait_stream(stream)

    def _offload_to_cpu(self, param: torch.Tensor, key: str, src: torch.Tensor) -> torch.Tensor:
        buffer = self._cpu_buffers.get((param, key))
        if buffer is None or buffer.shape != src.shape or buffer.dtype != src.dtype:
            buffer = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
            self._cpu_buffers[(param, key)] = buffer
        if buffer is not src:
            buffer.copy_(src, non_blocking=True)
        return buffer

    def _move_states(self, device: str):
        """Move optimizer states to CPU or back to GPU (matching each parameter's device)."""
        self.wait()
        for param in self.optimizer.state:
            state = self.optimizer.state[param]
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                local = _local(value)
                if device == "cpu":
                    new_local = self._offload_to_cpu(param, key, local)
                else:
                    new_local = local.to(device, non_blocking=True)
                state[key] = _with_local(value, new_local)
        if device == "cpu" and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def step(self, closure=None):
        # First step initializes states on GPU - offload after
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result

        # A parameter without state yet (no grad on any earlier step) is initialized by the full path below.
        if self._subset_step is not None and all(
            p in self.state for group in self.param_groups for p in group["params"] if p.grad is not None
        ):
            return self._chunked_step()

        # Move states to GPU
        self._move_states("cuda")

        # Run optimizer step
        result = self.optimizer.step(closure)

        # Move states back to CPU
        self._move_states("cpu")

        return result

    def _plan_chunks(
        self, units: list[list[torch.Tensor]]
    ) -> tuple[list[tuple[list[torch.Tensor], list[tuple[torch.Tensor, str, int]]]], int]:
        """Pack whole units into chunks; each streamed state gets an offset into a staging buffer.

        Chunk boundaries follow ``_nominal_bytes``, so every rank cuts the same chunks and a step that
        communicates runs the same collectives everywhere."""
        chunks, params, entries, used, nominal, max_used = [], [], [], 0, 0, 0
        for unit in units:
            unit_entries, unit_bytes, unit_nominal = [], 0, 0
            for param in unit:
                for key, value in self.optimizer.state[param].items():
                    if not isinstance(value, torch.Tensor) or value.ndim == 0:
                        continue
                    local = _local(value)
                    unit_entries.append((param, key, unit_bytes))
                    unit_bytes += -(-local.numel() * local.element_size() // ALIGN_BYTES) * ALIGN_BYTES
                    unit_nominal += _nominal_bytes(value)
            if params and nominal + unit_nominal > CHUNK_BYTES:
                chunks.append((params, entries))
                params, entries, used, nominal = [], [], 0, 0
            params.extend(unit)
            entries.extend((param, key, used + offset) for param, key, offset in unit_entries)
            used += unit_bytes
            nominal += unit_nominal
            max_used = max(max_used, used)
        if params:
            chunks.append((params, entries))
        return chunks, max_used

    def _chunked_step(self):
        """Step chunk by chunk (see ``_ChunkedStep``). Without an overlap model the chunks update on
        the compute stream and the step waits for the GPU; with one they update on a side stream in
        the forward order of their FSDP groups, enqueued as the next forward reaches them.

        Scalar step counters stay where they are (on CPU for torch AdamW), so the update never syncs
        the device."""
        self._finish_in_flight()
        if self._streams is None:
            self._streams = (torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream())
        h2d, d2h, side = self._streams
        subset_step = self._subset_step.begin_subset_step()
        units = subset_step.units
        if self._fsdp_groups:
            units = sorted(units, key=lambda unit: min(self._group_order[self._fsdp_groups[p]] for p in unit))
        chunks, buffer_bytes = self._plan_chunks(units)
        if not chunks:
            return
        if not self._fsdp_groups:
            step = _ChunkedStep(
                self.optimizer, subset_step, chunks, buffer_bytes, h2d, d2h, torch.cuda.current_stream()
            )
            step.advance(len(chunks))
            torch.cuda.synchronize()
            return
        self._in_flight = _ChunkedStep(self.optimizer, subset_step, chunks, buffer_bytes, h2d, d2h, side)
        for i, (params, _) in enumerate(chunks):
            for param in params:
                self._group_chunks[self._fsdp_groups[param]] = i
        self._in_flight.advance(INITIAL_CHUNKS - 1)

    def zero_grad(self, set_to_none: bool = True):
        if not set_to_none:
            # Zeroing in place would race the in-flight step's reads of the gradients.
            self.wait()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Move to GPU temporarily for consistent state dict
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()
        state_dict = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self._move_states("cpu")
        self._initialized = True

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    @property
    def base_optimizer(self) -> Optimizer:
        return self.optimizer

    def checkpoint_optimizer(self) -> Optimizer:
        return self.optimizer

    def prepare_checkpoint_save(self) -> None:
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()

    def finish_checkpoint_save(self) -> None:
        self._move_states("cpu")

    def finish_checkpoint_load(self) -> None:
        self._initialized = True
