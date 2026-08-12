import copy

import torch
from dion import Muon
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.configs.trainer import OptimizerConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.sign_sgd import SignSGD
from prime_rl.utils.logger import get_logger


class CPUOffloadOptimizer:
    """Wraps an optimizer to keep states on CPU, moving to GPU only for step().

    Unlike FSDP's CPUOffload which offloads weights too, this keeps weights on GPU.
    With activation checkpointing, activations and optimizer states are never on GPU
    at the same time: peak memory becomes max(activations, opt_states) instead of sum.
    """

    def __init__(self, optimizer: Optimizer, pin_memory: bool = True):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False

    def _move_states(self, device: str):
        """Move optimizer states to CPU or back to GPU (matching each parameter's device)."""
        for p in self.optimizer.state:
            state = self.optimizer.state[p]
            for k, v in state.items():
                if isinstance(v, DTensor):
                    local_tensor = v._local_tensor
                    if device == "cpu":
                        non_blocking = not self.pin_memory
                        new_local = local_tensor.to("cpu", non_blocking=non_blocking)
                        if self.pin_memory and not new_local.is_pinned():
                            new_local = new_local.pin_memory()
                    else:
                        new_local = local_tensor.to(device, non_blocking=True)
                    new_dtensor = copy.copy(v)
                    new_dtensor._local_tensor = new_local
                    state[k] = new_dtensor
                elif isinstance(v, torch.Tensor):
                    if v.dim() == 0:
                        # Scalar step counters stay on CPU: on CUDA the optimizer's
                        # per-parameter `.item()` read becomes a device sync.
                        if v.is_cuda:
                            state[k] = v.cpu()
                        continue
                    if device == "cpu":
                        non_blocking = not self.pin_memory
                        cpu_tensor = v.to("cpu", non_blocking=non_blocking)
                        if self.pin_memory and not cpu_tensor.is_pinned():
                            cpu_tensor = cpu_tensor.pin_memory()
                        state[k] = cpu_tensor
                    else:
                        state[k] = v.to(device, non_blocking=True)

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
        sd = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
        return sd

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


def setup_optimizer(
    config: OptimizerConfig,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    cpu_offload: bool = False,
) -> Optimizer | CPUOffloadOptimizer:
    optimizer = _create_optimizer(config, named_params, parallel_dims)

    if cpu_offload:
        get_logger().info("Wrapping optimizer with CPUOffloadOptimizer for optimizer state CPU offloading")
        return CPUOffloadOptimizer(optimizer)

    return optimizer


def _create_optimizer(
    config: OptimizerConfig,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    lr: float | None = None,
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
