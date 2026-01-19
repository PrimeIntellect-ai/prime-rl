import time
from typing import Callable

from dion import Muon
from torch import nn
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.trainer.config import OptimizerConfigType
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.runs import get_runs
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger


def setup_optimizer(
    config: OptimizerConfigType,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    lora: bool = False,
) -> Optimizer:
    if lora:
        # Wait for run 0 to be created in the runs system
        # Otherwise, the creation will reset the parameters
        runs = get_runs()
        world = get_world()
        logger = get_logger()
        while 0 not in runs.idx_2_id:
            if world.is_master:
                runs.check_for_changes()
            runs.sync_runs()
            logger.info(f"Waiting for run 0 to be created {runs.id_2_idx=}")
            time.sleep(1)
        named_params = runs.get_named_parameters_for_run(0)

    return _create_optimizer(config, named_params, parallel_dims)


def _create_optimizer(
    config: OptimizerConfigType,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    lr: float | None = None,
) -> Optimizer:
    """Create optimizer. If lr is None, uses config.lr."""
    if lr is None:
        lr = config.lr
    match config.type:
        case "sgd":
            return SGD(
                params=[p for _, p in named_params],
                lr=lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
        case "adamw":
            return AdamW(
                params=[p for _, p in named_params],
                lr=lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
            )
        case "muon":
            return _create_muon_optimizer(config, named_params, parallel_dims, lr)


def _create_muon_optimizer(
    config: OptimizerConfigType,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    lr: float | None = None,
) -> Optimizer:
    def muon_enabled(n, p):
        if p.ndim < 2:
            return False
        if "lm_head" in n:
            return False
        if "embed_tokens" in n:
            return False
        return True

    muon_params = []
    expert_params = []
    router_params = []
    adamw_params = []
    for n, p in named_params:
        if p.requires_grad and muon_enabled(n, p):
            if "mlp.experts" in n:
                expert_params.append(p)
            elif "mlp.router" in n:
                router_params.append(p)
            else:
                muon_params.append(p)
        elif p.requires_grad:
            adamw_params.append(p)
        else:
            pass

    param_groups = []

    param_groups.append(
        dict(params=muon_params, algorithm="muon", lr=lr, weight_decay=config.weight_decay, adjust_lr="rms_norm")
    )
    if expert_params:
        experts_mesh_name = None
        if parallel_dims.ep_enabled:
            experts_mesh_name = "dp_shard_mod_ep"
        param_groups.append(
            dict(
                params=expert_params,
                algorithm="muon",
                lr=lr,
                weight_decay=config.weight_decay,
                adjust_lr="rms_norm",
                distributed_mesh_name=experts_mesh_name,
            )
        )
    if router_params:
        param_groups.append(
            dict(
                params=router_params,
                algorithm="muon",
                lr=lr,
                weight_decay=config.weight_decay,
                adjust_lr="rms_norm",
            )
        )

    param_groups.append(dict(params=adamw_params, algorithm="adamw", lr=lr, weight_decay=config.weight_decay))

    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        distributed_mesh = parallel_dims.world_mesh["dp_shard_cp"]
    else:
        distributed_mesh = parallel_dims.world_mesh

    optimizer = Muon(
        params=param_groups,
        lr=lr,
        weight_decay=config.weight_decay,
        adjust_lr="rms_norm",
        distributed_mesh=distributed_mesh,
        world_mesh=parallel_dims.world_mesh,
        fsdp_mesh_dim=1 if parallel_dims.dp_replicate_enabled else 0,
    )
    return optimizer


class MultiLoRAOptimizer:
    def __init__(self, config: OptimizerConfigType, parallel_dims: ParallelDims):
        self.config = config
        self.parallel_dims = parallel_dims
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

        lr = self.runs.config[idx].optim.lr
        self.optimizers[idx] = _create_optimizer(self.config, named_params, self.parallel_dims, lr)

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


def setup_multi_optimizer(config: OptimizerConfigType, parallel_dims: ParallelDims) -> MultiLoRAOptimizer:
    return MultiLoRAOptimizer(config, parallel_dims)
