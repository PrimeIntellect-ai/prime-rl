import copy

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from prime_rl.trainer.optim.base import OffloadOptimizer


class CPUOffloadOptimizer(OffloadOptimizer):
    """Keep optimizer states on CPU between steps.

    Each state moves to the device of its parameter for the optimizer step. This
    supports models containing both GPU-resident and CPU-resident parameters.
    """

    def __init__(self, optimizer: Optimizer, pin_memory: bool = True):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False

    def _move_states(self, device: str | None):
        """Move optimizer states to CPU or back to GPU (matching each parameter's device)."""
        for param in self.optimizer.state:
            state = self.optimizer.state[param]
            if device is None:
                target_device = param.to_local().device if isinstance(param, DTensor) else param.device
            else:
                target_device = torch.device(device)
            for key, value in state.items():
                if isinstance(value, DTensor):
                    local_tensor = value._local_tensor
                    if target_device.type == "cpu":
                        non_blocking = not self.pin_memory
                        new_local = local_tensor.to("cpu", non_blocking=non_blocking)
                        if self.pin_memory and not new_local.is_pinned():
                            new_local = new_local.pin_memory()
                    else:
                        new_local = local_tensor.to(target_device, non_blocking=True)
                    new_dtensor = copy.copy(value)
                    new_dtensor._local_tensor = new_local
                    state[key] = new_dtensor
                elif isinstance(value, torch.Tensor):
                    if target_device.type == "cpu":
                        non_blocking = not self.pin_memory
                        cpu_tensor = value.to("cpu", non_blocking=non_blocking)
                        if self.pin_memory and not cpu_tensor.is_pinned():
                            cpu_tensor = cpu_tensor.pin_memory()
                        state[key] = cpu_tensor
                    else:
                        state[key] = value.to(target_device, non_blocking=True)

    def step(self, closure=None):
        # The first step initializes each state beside its parameter.
        if not self._initialized:
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            self._initialized = True
            return result

        # Move each state back to its parameter's device.
        self._move_states(None)

        # Run optimizer step
        result = self.optimizer.step(closure)

        # Move states back to CPU
        self._move_states("cpu")

        return result

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Move states to their parameters temporarily for a consistent state dict.
        if self._initialized:
            self._move_states(None)
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
            self._move_states(None)
            torch.cuda.synchronize()

    def finish_checkpoint_save(self) -> None:
        self._move_states("cpu")

    def finish_checkpoint_load(self) -> None:
        self._initialized = True
