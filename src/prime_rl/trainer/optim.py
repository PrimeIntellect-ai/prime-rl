import copy
import re
from typing import Callable

import torch
from dion import Muon
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.configs.trainer import OptimizerConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.sign_sgd import SignSGD
from prime_rl.utils.logger import get_logger


class CPUOffloadOptimizer:
    """Keeps optimizer states on CPU, moving them to GPU only for step().

    Weights stay on GPU (unlike FSDP CPUOffload). With activation checkpointing,
    activations and optimizer states are never on GPU at the same time, so peak
    memory is max(activations, opt_states) instead of sum.

    When ``named_params`` is provided, the step is performed per-transformer-layer
    ("chunked") instead of all-at-once, bounding peak GPU optimizer-state memory to
    one layer's worth. H2D/D2H transfers can optionally overlap with compute on
    dedicated CUDA streams (``stream=True``).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        named_params: list[tuple[str, nn.Parameter]] | None = None,
        pin_memory: bool = True,
        stream: bool = False,
    ):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self.stream = stream
        self._initialized = False
        self._chunks: list[list[nn.Parameter]] | None = None
        if named_params is not None:
            self._chunks = self._build_chunks(named_params)

    @staticmethod
    def _extract_layer_idx(name: str) -> int | None:
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def _build_chunks(self, named_params: list[tuple[str, nn.Parameter]]) -> list[list[nn.Parameter]]:
        """Group optimizer params by transformer layer; non-layer params go last."""
        param_layer = {id(p): self._extract_layer_idx(n) for n, p in named_params}
        by_layer: dict[int, list[nn.Parameter]] = {}
        misc: list[nn.Parameter] = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                idx = param_layer.get(id(p))
                if idx is not None:
                    by_layer.setdefault(idx, []).append(p)
                else:
                    misc.append(p)
        chunks = [by_layer[k] for k in sorted(by_layer)]
        if misc:
            chunks.append(misc)
        return chunks or [misc]

    def _move_tensor(self, v: torch.Tensor, device: str) -> torch.Tensor:
        if device == "cpu":
            if self.pin_memory:
                dst = torch.empty_like(v, device="cpu").pin_memory()
                dst.copy_(v, non_blocking=True)
                return dst
            return v.to("cpu")
        return v.to(device, non_blocking=True)

    def _move_states(self, device: str):
        """Move all optimizer states to *device*."""
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, DTensor):
                    new_local = self._move_tensor(v._local_tensor, device)
                    new_dtensor = copy.copy(v)
                    new_dtensor._local_tensor = new_local
                    state[k] = new_dtensor
                elif isinstance(v, torch.Tensor):
                    state[k] = self._move_tensor(v, device)

    def _move_chunk_states(self, chunk_idx: int, device: str, stream: torch.cuda.Stream | None = None):
        """Move optimizer states for a single chunk to/from *device*.

        When *stream* is provided the transfers are issued on that stream. For D2H
        (device == "cpu"), the old GPU tensor is recorded on the stream so the
        caching allocator won't reuse its storage until the async copy finishes.
        """
        chunk_ids = {id(p) for p in self._chunks[chunk_idx]}
        ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.default_stream()
        with ctx:
            for p, state in self.optimizer.state.items():
                if id(p) not in chunk_ids:
                    continue
                for k, v in state.items():
                    if isinstance(v, DTensor):
                        if device == "cpu" and stream is not None and v._local_tensor.is_cuda:
                            v._local_tensor.record_stream(stream)
                        new_local = self._move_tensor(v._local_tensor, device)
                        new_dtensor = copy.copy(v)
                        new_dtensor._local_tensor = new_local
                        state[k] = new_dtensor
                    elif isinstance(v, torch.Tensor):
                        if device == "cpu" and stream is not None and v.is_cuda:
                            v.record_stream(stream)
                        state[k] = self._move_tensor(v, device)

    def _step_chunk(self, chunk_idx: int, closure=None):
        """Run optimizer.step() for a single chunk by temporarily swapping param_groups."""
        chunk_ids = {id(p) for p in self._chunks[chunk_idx]}
        chunk_groups = []
        for group in self._original_param_groups:
            filtered = [p for p in group["params"] if id(p) in chunk_ids]
            if not filtered:
                continue
            new_group = {k: v for k, v in group.items() if k != "params"}
            new_group["params"] = filtered
            chunk_groups.append(new_group)
        self.optimizer.param_groups = chunk_groups
        result = self.optimizer.step(closure)
        self.optimizer.param_groups = self._original_param_groups
        return result

    def _sync_step_counters(self, original_steps: list):
        """Increment the original param_groups' step counters by one.

        Muon stores a per-group ``step`` that ``step()`` increments. Because we swap
        in per-chunk copies, the original groups never see the increment. Standard
        optimizers (AdamW, SGD, SignSGD) keep step in ``state[p]['step']`` which is
        per-parameter and already correct.
        """
        for group, orig_step in zip(self._original_param_groups, original_steps):
            group["step"] = orig_step + 1

    def step(self, closure=None):
        # Non-chunked path.
        if self._chunks is None or len(self._chunks) <= 1:
            if not self._initialized:
                result = self.optimizer.step(closure)
                self._move_states("cpu")
                torch.cuda.synchronize()
                self._initialized = True
                return result
            self._move_states("cuda")
            result = self.optimizer.step(closure)
            self._move_states("cpu")
            return result

        # Chunked path (also used for the first step to avoid materializing
        # all optimizer states on GPU at once during initialization).
        self._original_param_groups = self.optimizer.param_groups
        original_steps = [g.get("step", 0) for g in self._original_param_groups]

        if self.stream:
            self._step_chunked_streamed(closure)
        else:
            self._step_chunked(closure)

        self._sync_step_counters(original_steps)
        self._original_param_groups = None
        self._initialized = True
        return None

    def _step_chunked(self, closure=None):
        """Sequential per-layer step without stream overlap."""
        for i in range(len(self._chunks)):
            self._move_chunk_states(i, "cuda")
            self._step_chunk(i, closure if i == 0 else None)
            self._move_chunk_states(i, "cpu")
        torch.cuda.synchronize()

    def _step_chunked_streamed(self, closure=None):
        """Per-layer step with stream-overlapped H2D / D2H."""
        h2d_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.current_stream()
        n = len(self._chunks)

        self._move_chunk_states(0, "cuda", h2d_stream)

        for i in range(n):
            compute_stream.wait_stream(h2d_stream)

            if i + 1 < n:
                # Wait for previous D2H before reusing pinned CPU buffers.
                h2d_stream.wait_stream(d2h_stream)
                self._move_chunk_states(i + 1, "cuda", h2d_stream)

            self._step_chunk(i, closure if i == 0 else None)

            d2h_stream.wait_stream(compute_stream)
            self._move_chunk_states(i, "cpu", d2h_stream)

        torch.cuda.synchronize()

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        if self._initialized:
            self._move_states("cuda")
            torch.cuda.synchronize()
        sd = self.optimizer.state_dict()
        if self._initialized:
            self._move_states("cpu")
            torch.cuda.synchronize()
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
    lora: bool = False,
    cpu_offload: bool = False,
    cpu_offload_chunked: bool = False,
    cpu_offload_stream: bool = False,
) -> Optimizer | CPUOffloadOptimizer:
    if lora:
        # Wait for run 0 to be created in the multi run manager
        # Otherwise, the creation will reset the parameters
        multi_run_manager = get_multi_run_manager()
        multi_run_manager.wait_for_run(0)
        named_params = multi_run_manager.get_named_parameters_for_run(0)

    optimizer = _create_optimizer(config, named_params, parallel_dims)

    if cpu_offload:
        get_logger().info("Wrapping optimizer with CPUOffloadOptimizer for optimizer state CPU offloading")
        return CPUOffloadOptimizer(
            optimizer,
            named_params=named_params if cpu_offload_chunked else None,
            stream=cpu_offload_stream,
        )

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


class MultiLoRAOptimizer:
    def __init__(self, config: OptimizerConfig, parallel_dims: ParallelDims):
        self.config = config
        self.parallel_dims = parallel_dims
        self.multi_run_manager = get_multi_run_manager()
        self.logger = get_logger()

        self.optimizers: list[Optimizer | None] = [None] * self.multi_run_manager.max_runs
        self._post_creation_callbacks: list[Callable[[Optimizer, int], None]] = []

        # Register creation hook for optimizer setup
        # The MultiRunManager class handles parameter reset internally when new runs are created
        self.multi_run_manager.register_creation_hook(self.optimizer_creation_hook)

    def register_post_creation_callback(
        self, callback: Callable[[Optimizer, int], None], index: int | None = None
    ) -> None:
        """Register a callback to be called after an optimizer is created.

        Args:
            callback: A callable that takes (optimizer: Optimizer, idx: int) as arguments.
            index: Optional index to insert at. If None, appends to end.
        """
        if index is None:
            self._post_creation_callbacks.append(callback)
        else:
            self._post_creation_callbacks.insert(index, callback)

    def optimizer_creation_hook(self, idx: int, run_id: str) -> None:
        # Get named parameters for this run from the MultiRunManager system
        named_params = self.multi_run_manager.get_named_parameters_for_run(idx)

        lr = self.multi_run_manager.config[idx].optim.lr
        self.optimizers[idx] = _create_optimizer(self.config, named_params, self.parallel_dims, lr)

        # Call post-creation callbacks in order
        for callback in self._post_creation_callbacks:
            callback(self.optimizers[idx], idx)

    def step(self):
        for idx in self.multi_run_manager.ready_to_update_idxs:
            self.optimizers[idx].step()

    def zero_grad(self):
        for idx in self.multi_run_manager.ready_to_update_idxs:
            self.optimizers[idx].zero_grad()

    def get_current_lr(self, idx: int | None = None) -> float:
        if idx is None:
            for idx in self.multi_run_manager.ready_to_update_idxs:
                return self.optimizers[idx].param_groups[0]["lr"]
            else:
                self.logger.warning("No runs are ready to update. Returning 0.0 for current learning rate.")
                return 0.0
        else:
            return self.optimizers[idx].param_groups[0]["lr"]


def setup_multi_optimizer(config: OptimizerConfig, parallel_dims: ParallelDims) -> MultiLoRAOptimizer:
    return MultiLoRAOptimizer(config, parallel_dims)
