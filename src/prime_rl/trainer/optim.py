from dion import Muon
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.trainer.config import OptimizerConfigType
from prime_rl.trainer.runs import get_runs
from prime_rl.utils.logger import get_logger


class MultiOptimizer:
    def __init__(self, config: OptimizerConfigType, model: nn.Module, device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.runs = get_runs()
        self.logger = get_logger()

        self.optimizers: list[Optimizer | None] = [None] * self.runs.max_runs

        # Register creation hook for optimizer setup
        # The Runs class handles parameter reset internally when new runs are created
        self.runs.register_creation_hook(self.optimizer_creation_hook)

    def optimizer_creation_hook(self, idx: int, run_id: str) -> None:
        # Get named parameters for this run from the Runs system
        named_params = self.runs.get_named_parameters_for_run(idx)
        self.optimizers[idx] = _setup_optimizer(self.config, named_params, self.device_mesh)

    def step(self):
        for idx in self.runs.used_idxs:
            print(f"Stepping optimizer for run {idx}")
            self.optimizers[idx].step()

            # except Exception as e:
            #    self.logger.error(f"Error stepping optimizer for run {idx}: {e}")

    def zero_grad(self):
        for idx in self.runs.used_idxs:
            try:
                self.optimizers[idx].zero_grad()
            except Exception as e:
                self.logger.error(f"Error zeroing grad for run {idx}: {e}")

    def state_dict(self):
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict):
        for optimizer, optimizer_state in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(optimizer_state)


def setup_multi_optimizer(config: OptimizerConfigType, model: nn.Module, device_mesh: DeviceMesh) -> MultiOptimizer:
    return MultiOptimizer(config, model, device_mesh)


def setup_optimizer(config: OptimizerConfigType, model: nn.Module, device_mesh: DeviceMesh) -> Optimizer:
    return _setup_optimizer(config, list(model.named_parameters()), device_mesh)


def _setup_optimizer(
    config: OptimizerConfigType, named_params: list[tuple[str, nn.Parameter]], device_mesh: DeviceMesh
) -> Optimizer:
    match config.type:
        case "sgd":
            return SGD(
                params=[p for n, p in named_params],
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
        case "adamw":
            return AdamW(
                params=[p for n, p in named_params],
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
