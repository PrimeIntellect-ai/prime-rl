import copy

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from prime_rl.trainer.optim.base import OffloadOptimizer


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

        # Move states to GPU
        self._move_states("cuda")

        # Run optimizer step
        result = self.optimizer.step(closure)

        # Move states back to CPU
        self._move_states("cpu")

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
