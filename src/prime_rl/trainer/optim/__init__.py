from dion import Muon
from torch import nn
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.configs.trainer import FullOptimizerOffloadingConfig, OptimizerConfig
from prime_rl.trainer.optim.offload import (
    FullCPUOffloadOptimizer,
    GradientOffloadManager,
    _create_cpu_master_weights,
)
from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.sign_sgd import SignSGD
from prime_rl.utils.logger import get_logger


def setup_optimizer(
    config: OptimizerConfig,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    cpu_offload: bool = False,
    full_offload_config: FullOptimizerOffloadingConfig | None = None,
    model: nn.Module | None = None,
) -> tuple[Optimizer | CPUOffloadOptimizer | FullCPUOffloadOptimizer, GradientOffloadManager | None]:
    if cpu_offload and full_offload_config is not None:
        raise ValueError("State-only and full optimizer CPU offload cannot both be enabled")
    if full_offload_config is not None and config.type == "muon":
        raise ValueError("CPU optimizer offload does not support Muon because the optimizer step runs on CPU")
    if full_offload_config is not None and config.max_norm is not None:
        get_logger().warning("Disabling gradient clipping because CPU optimizer offload updates during backward")
        config.max_norm = None
    optimizer_named_params = named_params
    master_weights = None
    if full_offload_config is not None:
        if model is None:
            raise ValueError("CPU optimizer offload requires the model")
        optimizer_named_params, master_weights = _create_cpu_master_weights(
            model,
            named_params,
            pin_memory=not (config.type == "adamw" and full_offload_config.cpu_optimizer_backend == "native"),
        )

    optimizer = _create_optimizer(
        config,
        optimizer_named_params,
        parallel_dims,
        fused_adamw=full_offload_config is not None and config.type == "adamw",
    )

    if full_offload_config is not None:
        assert master_weights is not None
        get_logger().info("Using CPU offload for gradients and the optimizer step")
        optimizer = FullCPUOffloadOptimizer(
            optimizer,
            offload_config=full_offload_config,
            master_weights=master_weights,
            dp_replicate=parallel_dims.dp_replicate,
        )
        return optimizer, optimizer._gradient_manager

    if cpu_offload:
        get_logger().info("Wrapping optimizer with CPUOffloadOptimizer for optimizer state CPU offloading")
        return CPUOffloadOptimizer(optimizer), None

    return optimizer, None


def _create_optimizer(
    config: OptimizerConfig,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    lr: float | None = None,
    fused_adamw: bool = False,
) -> Optimizer:
    """Create optimizer. If lr is None, uses config.lr."""
    if lr is None:
        lr = config.lr
    # Only hand trainable params to the optimizer. Frozen params (e.g. the DSA sparse
    # indexer, which runs under no_grad) carry no optimizer state, and including them
    # breaks strict checkpoint resume (DCP materializes state for every requires_grad
    # param at load time, mismatching the saved state). Muon filters internally below.
    trainable_params = [p for _, p in named_params if p.requires_grad]
    match config.type:
        case "sgd":
            return SGD(
                params=trainable_params,
                lr=lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
        case "adamw":
            return AdamW(
                params=trainable_params,
                lr=lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
                fused=fused_adamw,
            )
        case "muon":
            return _create_muon_optimizer(config, named_params, parallel_dims, lr)
        case "sign_sgd":
            return SignSGD(
                params=trainable_params,
                lr=lr,
                weight_decay=config.weight_decay,
            )


def _create_muon_optimizer(
    config: OptimizerConfig,
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
        distributed_mesh = parallel_dims.get_mesh("dp_shard_cp")
    else:
        distributed_mesh = parallel_dims.world_mesh

    optimizer = Muon(
        params=param_groups,
        lr=lr,
        mu=config.mu,
        betas=(config.betas1, config.betas2),
        weight_decay=config.weight_decay,
        adjust_lr="rms_norm",
        distributed_mesh=distributed_mesh,
        world_mesh=parallel_dims.world_mesh,
        fsdp_mesh_dim=1 if parallel_dims.dp_replicate_enabled else 0,
    )
    return optimizer
