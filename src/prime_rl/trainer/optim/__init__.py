import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from prime_rl.configs.trainer import OffloadMode, OptimizerConfig
from prime_rl.trainer.models.fusions import get_model_packed_parameters
from prime_rl.trainer.models.layers.ngram import NGramTable
from prime_rl.trainer.optim.base import OffloadOptimizer as OffloadOptimizer
from prime_rl.trainer.optim.base import OptimizerLike
from prime_rl.trainer.optim.muon import Muon
from prime_rl.trainer.optim.offload import (
    FullCPUOffloadOptimizer,
    GradientOffloadManager,
    _create_cpu_master_weights,
)
from prime_rl.trainer.optim.sign_sgd import SignSGD
from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.logger import get_logger


def setup_optimizer(
    config: OptimizerConfig,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    offload: OffloadMode = "none",
    model: nn.Module | None = None,
    full_offload_dtype_policy: dict[int, tuple[torch.dtype, torch.dtype]] | None = None,
) -> tuple[OptimizerLike, GradientOffloadManager | None]:
    cpu_offload = offload == "optimizer"
    full_offload = offload == "full"
    if full_offload and config.type not in ("adamw", "sign_sgd"):
        raise ValueError("Full optimizer offload only supports AdamW and SignSGD")
    if full_offload and config.max_norm is not None:
        get_logger().warning("Disabling gradient clipping because CPU optimizer offload updates during backward")
        config.max_norm = None
    optimizer_named_params = named_params
    master_weights = None
    if full_offload:
        if model is None:
            raise ValueError("CPU optimizer offload requires the model")
        if full_offload_dtype_policy is None:
            raise ValueError("CPU optimizer offload requires an explicit per-parameter dtype policy")
        optimizer_named_params, master_weights = _create_cpu_master_weights(
            model, named_params, dtype_policy=full_offload_dtype_policy
        )

    optimizer = _create_optimizer(
        config,
        optimizer_named_params,
        fused_adamw=config.type == "adamw" and not cpu_offload,
        model=model,
    )

    if full_offload:
        assert master_weights is not None
        get_logger().info("Using CPU offload for gradients and the optimizer step")
        optimizer = FullCPUOffloadOptimizer(
            optimizer,
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
    lr: float | None = None,
    fused_adamw: bool = False,
    model: nn.Module | None = None,
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
        case "adamw":
            return AdamW(
                params=trainable_params,
                lr=lr,
                weight_decay=config.weight_decay,
                betas=(config.betas1, config.betas2),
                eps=config.eps,
                fused=fused_adamw,
            )
        case "muon":
            return _create_muon_optimizer(config, named_params, model, lr)
        case "sign_sgd":
            return SignSGD(
                params=trainable_params,
                lr=lr,
                weight_decay=config.weight_decay,
            )


def _create_muon_optimizer(
    config: OptimizerConfig,
    named_params: list[tuple[str, nn.Parameter]],
    model: nn.Module | None,
    lr: float | None = None,
) -> Optimizer:
    ngram_table_params = set()
    if model is not None:
        ngram_table_params = {
            p for module in model.modules() if isinstance(module, NGramTable) for p in module.parameters()
        }

    def muon_enabled(n, p):
        if p.ndim < 2:
            return False
        if "lm_head" in n:
            return False
        if "embed_tokens" in n:
            return False
        if p in ngram_table_params:
            return False
        # Expert biases should not go to Muon (and would also raise under EP).
        if "mlp.experts" in n and n.endswith("_bias"):
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

    # dion's groups and keys, so that checkpoints of runs on dion's Muon load as they are
    param_groups = [dict(params=muon_params, algorithm="muon")]
    if expert_params:
        param_groups.append(dict(params=expert_params, algorithm="muon"))
    if router_params:
        param_groups.append(dict(params=router_params, algorithm="muon"))
    param_groups.append(dict(params=adamw_params, algorithm="adamw"))

    # Runtime fusions pack several logical matrices into one physical parameter. Muon
    # orthogonalizes each of them separately, so a packed parameter trains exactly as the
    # matrices it replaces would. Packed biases and frozen parameters are not Muon's.
    matrix_partitions = {}
    if model is not None:
        muon_params = {p for group in param_groups if group["algorithm"] == "muon" for p in group["params"]}
        for packed_info in get_model_packed_parameters(model):
            partitions = packed_info.spec.muon_matrix_partitions(packed_info.parameter)
            if partitions is not None and packed_info.parameter in muon_params:
                matrix_partitions[packed_info.parameter] = partitions

    return Muon(
        param_groups,
        lr=lr,
        mu=config.mu,
        betas=(config.betas1, config.betas2),
        weight_decay=config.weight_decay,
        epsilon=config.eps,
        nesterov=config.nesterov,
        update_rms=config.update_rms,
        newton_schulz=config.newton_schulz,
        matrix_partitions=matrix_partitions,
    )
