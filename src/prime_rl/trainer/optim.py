from typing import Callable

from dion import Muon
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.trainer.config import OptimizerConfigType
from prime_rl.trainer.runs import get_runs
from prime_rl.utils.logger import get_logger


def setup_optimizer(
    config: OptimizerConfigType, named_params: list[tuple[str, nn.Parameter]], device_mesh: DeviceMesh
) -> Optimizer:
    match config.type:
        case "sgd":
            return SGD(
                params=[p for _, p in named_params],
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
        case "adamw":
            return AdamW(
                params=[p for _, p in named_params],
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
            )
        case "muon":

            def muon_enabled(n, p):
                if p.ndim < 2:
                    return False
                if "lm_head" in n:
                    return False
                if "embed_tokens" in n:
                    return False
                return True

            muon_params = [p for n, p in named_params if p.requires_grad and muon_enabled(n, p)]
            adamw_params = [p for n, p in named_params if p.requires_grad and not muon_enabled(n, p)]

            optimizer = Muon(
                [
                    dict(
                        params=muon_params,
                        algorithm="muon",
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                        adjust_lr="rms_norm",
                    ),
                    dict(params=adamw_params, algorithm="adamw", lr=config.lr, weight_decay=config.weight_decay),
                ],
                lr=config.lr,
                weight_decay=config.weight_decay,
                adjust_lr="rms_norm",
                distributed_mesh=device_mesh,
            )

            return optimizer


class MultiOptimizer:
    def __init__(self, config: OptimizerConfigType, device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.runs = get_runs()
        self.logger = get_logger()

        self.optimizers: list[Optimizer | None] = [None] * self.runs.max_runs
        self._post_creation_callbacks: list[Callable[[Optimizer, int], None]] = []

        # Register creation hook for optimizer setup
        # The Runs class handles parameter reset internally when new runs are created
        self.runs.register_creation_hook(self.optimizer_creation_hook)

    def register_post_creation_callback(self, callback: Callable[[Optimizer, int], None]) -> None:
        """Register a callback to be called after an optimizer is created.

        Args:
            callback: A callable that takes (optimizer: Optimizer, idx: int) as arguments.
        """
        self._post_creation_callbacks.append(callback)

    def optimizer_creation_hook(self, idx: int, run_id: str) -> None:
        # Get named parameters for this run from the Runs system
        named_params = self.runs.get_named_parameters_for_run(idx)
        self.optimizers[idx] = setup_optimizer(self.config, named_params, self.device_mesh)

        # Call post-creation callbacks (e.g., for scheduler creation)
        for callback in self._post_creation_callbacks:
            callback(self.optimizers[idx], idx)

    def step(self):
        for idx in self.runs.ready_to_update_idxs:
            self.optimizers[idx].step()

    def zero_grad(self):
        for idx in self.runs.ready_to_update_idxs:
            self.optimizers[idx].zero_grad()

    def get_current_lr(self, idx: int | None = None) -> float:
        if idx is None:
            for idx in self.runs.ready_to_update_idxs:
                return self.optimizers[idx].param_groups[0]["lr"]
            else:
                self.logger.warning("No runs are ready to update. Returning 0.0 for current learning rate.")
                return 0.0
        else:
            return self.optimizers[idx].param_groups[0]["lr"]


def setup_multi_optimizer(config: OptimizerConfigType, device_mesh: DeviceMesh) -> MultiOptimizer:
    return MultiOptimizer(config, device_mesh)
