from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TypeAlias

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


class OffloadOptimizer(ABC):
    @property
    @abstractmethod
    def base_optimizer(self) -> Optimizer: ...

    @abstractmethod
    def checkpoint_optimizer(self) -> Optimizer: ...

    def prepare_checkpoint_save(self) -> None:
        pass

    def finish_checkpoint_save(self) -> None:
        pass

    @abstractmethod
    def finish_checkpoint_load(self) -> None: ...

    def finish_model_only_checkpoint_load(self) -> None:
        pass


OptimizerLike: TypeAlias = Optimizer | OffloadOptimizer


class GPUGradientManager:
    """Gradient scaling and clipping for gradients that stay on the GPU.

    Has the same step/backward interface as `GradientOffloadManager`. The step's gradient
    scale is applied once, after the final backward of the step.
    """

    def __init__(self, model: nn.Module):
        self._model = model
        self._gradient_scale = 1.0
        self._final_backward = False

    def begin_step(self, gradient_scale: float, *, overlap_optimizer: bool) -> None:
        self._gradient_scale = gradient_scale

    def begin_backward(self, *, final_backward: bool) -> None:
        self._final_backward = final_backward

    @torch.no_grad()
    def finish_backward(self) -> None:
        if not self._final_backward:
            return
        for param in self._model.parameters():
            if param.grad is not None:
                param.grad.mul_(self._gradient_scale)

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        # Norm reductions must complete on each parameter mesh before combining
        # dense, expert-parallel, and other model-parallel gradients.
        mesh_parameters = defaultdict(list)
        for param in self._model.parameters():
            if param.grad is not None:
                mesh = param.grad.device_mesh if isinstance(param.grad, DTensor) else None
                mesh_parameters[mesh].append(param)
        norms = []
        for parameters in mesh_parameters.values():
            norm = torch.nn.utils.get_total_norm([param.grad for param in parameters])
            norms.append(norm.full_tensor() if isinstance(norm, DTensor) else norm)
        grad_norm = torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.tensor(0.0)
        for parameters in mesh_parameters.values():
            torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, grad_norm)
        return grad_norm.cuda() if grad_norm.device.type == "cpu" else grad_norm

    def close(self) -> None:
        pass
