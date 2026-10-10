import copy

import torch
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer

from prime_rl.trainer.optim.base import OffloadOptimizer

# Byte budget of one staging chunk. A single state larger than this gets a chunk of its own.
CHUNK_BYTES = 1 << 30
# Alignment of each state view inside a staging buffer.
ALIGN_BYTES = 512


def _local(value: torch.Tensor) -> torch.Tensor:
    return value._local_tensor if isinstance(value, DTensor) else value


def _with_local(value: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, DTensor):
        return local
    new_dtensor = copy.copy(value)
    new_dtensor._local_tensor = local
    return new_dtensor


class CPUOffloadOptimizer(OffloadOptimizer):
    """Wraps an optimizer to keep states on CPU, moving to GPU only for step().

    Unlike FSDP's CPUOffload which offloads weights too, this keeps weights on GPU.
    With activation checkpointing, activations and optimizer states are never on GPU
    at the same time: peak memory becomes max(activations, opt_states) instead of sum.
    """

    def __init__(self, optimizer: Optimizer, pin_memory: bool = True):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False
        self._cpu_buffers: dict[tuple[torch.Tensor, str], torch.Tensor] = {}
        self._streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = None

    def _offload_to_cpu(self, param: torch.Tensor, key: str, src: torch.Tensor) -> torch.Tensor:
        if not self.pin_memory:
            return src.to("cpu", non_blocking=True)
        buffer = self._cpu_buffers.get((param, key))
        if buffer is None or buffer.shape != src.shape or buffer.dtype != src.dtype:
            buffer = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
            self._cpu_buffers[(param, key)] = buffer
        if buffer is not src:
            buffer.copy_(src, non_blocking=True)
        return buffer

    def _move_states(self, device: str):
        """Move optimizer states to CPU or back to GPU (matching each parameter's device)."""
        for param in self.optimizer.state:
            state = self.optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, DTensor):
                    local_tensor = value._local_tensor
                    if device == "cpu":
                        new_local = self._offload_to_cpu(param, key, local_tensor)
                    else:
                        new_local = local_tensor.to(device, non_blocking=True)
                    new_dtensor = copy.copy(value)
                    new_dtensor._local_tensor = new_local
                    state[key] = new_dtensor
                elif isinstance(value, torch.Tensor):
                    if device == "cpu":
                        state[key] = self._offload_to_cpu(param, key, value)
                    else:
                        state[key] = value.to(device, non_blocking=True)
        if device == "cpu" and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def step(self, closure=None):
        # First step initializes states on GPU - offload after
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result

        if closure is None and self.pin_memory and isinstance(self.optimizer, AdamW):
            return self._chunked_step()

        # Move states to GPU
        self._move_states("cuda")

        # Run optimizer step
        result = self.optimizer.step(closure)

        # Move states back to CPU
        self._move_states("cpu")

        return result

    def _plan_chunks(self) -> tuple[list[tuple[set[torch.Tensor], list[tuple[torch.Tensor, str, int]]]], int]:
        """Group whole parameters into chunks; each state except the CPU step counter gets an offset into a staging buffer."""
        chunks, params, entries, used, max_used = [], set(), [], 0, 0
        for param, state in self.optimizer.state.items():
            param_entries, param_bytes = [], 0
            for key, value in state.items():
                if key == "step":
                    continue
                local = _local(value)
                param_entries.append((param, key, param_bytes))
                param_bytes += -(-local.numel() * local.element_size() // ALIGN_BYTES) * ALIGN_BYTES
            if params and used + param_bytes > CHUNK_BYTES:
                chunks.append((params, entries))
                params, entries, used = set(), [], 0
            params.add(param)
            entries.extend((param, key, used + offset) for param, key, offset in param_entries)
            used += param_bytes
            max_used = max(max_used, used)
        if params:
            chunks.append((params, entries))
        return chunks, max_used

    def _step_params(self, params: set[torch.Tensor]):
        groups = self.optimizer.param_groups
        all_params = [group["params"] for group in groups]
        for group in groups:
            group["params"] = [param for param in group["params"] if param in params]
        try:
            return self.optimizer.step()
        finally:
            for group, group_params in zip(groups, all_params):
                group["params"] = group_params

    def _chunked_step(self):
        """Step chunk by chunk through two staging buffers: while chunk i updates on the compute stream,
        chunk i+1 loads on one side stream and chunk i-1 writes back on another.

        The step counters stay on CPU, as in plain torch AdamW, so the update never syncs the device."""
        if self._streams is None:
            self._streams = (torch.cuda.Stream(), torch.cuda.Stream())
        h2d, d2h = self._streams
        compute = torch.cuda.current_stream()
        chunks, buffer_bytes = self._plan_chunks()
        state = self.optimizer.state
        # Allocated on the compute stream, so the caching allocator reuses the same blocks every step.
        buffers = [torch.empty(buffer_bytes, dtype=torch.uint8, device="cuda") for _ in range(2)]
        h2d.wait_stream(compute)
        written_back: list[torch.cuda.Event | None] = [None, None]
        cpu_values: list[dict[tuple[torch.Tensor, str], torch.Tensor]] = [{}, {}]

        def load(i: int) -> torch.cuda.Event:
            buffer, slot = buffers[i % 2], cpu_values[i % 2]
            with torch.cuda.stream(h2d):
                if written_back[i % 2] is not None:
                    h2d.wait_event(written_back[i % 2])
                slot.clear()
                for param, key, offset in chunks[i][1]:
                    cpu_value = state[param][key]
                    cpu_local = _local(cpu_value)
                    nbytes = cpu_local.numel() * cpu_local.element_size()
                    gpu_local = buffer[offset : offset + nbytes].view(cpu_local.dtype).view(cpu_local.shape)
                    gpu_local.copy_(cpu_local, non_blocking=True)
                    slot[param, key] = cpu_value
                    state[param][key] = _with_local(cpu_value, gpu_local)
                return h2d.record_event()

        result = None
        loaded = load(0) if chunks else None
        for i, (params, _) in enumerate(chunks):
            next_loaded = load(i + 1) if i + 1 < len(chunks) else None
            compute.wait_event(loaded)
            result = self._step_params(params)
            d2h.wait_stream(compute)
            with torch.cuda.stream(d2h):
                for (param, key), cpu_value in cpu_values[i % 2].items():
                    _local(cpu_value).copy_(_local(state[param][key]), non_blocking=True)
                    state[param][key] = cpu_value
                written_back[i % 2] = d2h.record_event()
            loaded = next_loaded
        if not chunks:
            result = self.optimizer.step()
        compute.wait_stream(d2h)
        torch.cuda.synchronize()
        return result

    def zero_grad(self, set_to_none: bool = True):
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
