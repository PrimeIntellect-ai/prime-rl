import copy
import queue
import re
import threading
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
from dion import Muon
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.configs.trainer import OptimizerConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.sign_sgd import SignSGD
from prime_rl.utils.logger import get_logger


@dataclass
class _CPUGradientBuffer:
    template: DTensor
    accumulator: torch.Tensor
    staging: torch.Tensor | None = None
    initialized: bool = False
    pending: bool = False


@dataclass
class _GradientCopyTask:
    event: torch.cuda.Event
    buffers: list[tuple[_CPUGradientBuffer, bool]]


class GradientOffloadManager:
    """Evicts finalized FSDP2 sharded gradients while backward is still running."""

    def __init__(self, model: nn.Module, chunks: list[list[nn.Parameter]], dp_replicate: int):
        if not torch.__version__.startswith("2.11."):
            raise RuntimeError(
                "Gradient CPU offload hook ordering is verified only for the locked PyTorch 2.11 release"
            )
        self._chunks = chunks
        self._dp_replicate = dp_replicate
        self._optimizer_param_ids = {id(param) for chunk in chunks for param in chunk}
        self._buffers: dict[int, _CPUGradientBuffer] = {}

        self._d2h_stream = torch.cuda.Stream()
        self._tasks: queue.SimpleQueue[_GradientCopyTask] = queue.SimpleQueue()
        self._condition = threading.Condition()

        self._lagged_unit: tuple[list[nn.Parameter], object | None] | None = None
        self._seen_units: set[int] = set()

        self._gradient_scale = 1.0
        self._logged_cpu_allocation = 0

        fsdp_modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
        if not fsdp_modules:
            raise ValueError("Gradient CPU offload requires an FSDP2 model")
        for module in fsdp_modules:
            module.register_full_backward_hook(self._backward_hook)

        threading.Thread(target=self._copy_worker, name="grad-offload", daemon=True).start()
        get_logger().info(
            "Gradient CPU offload uses the lag-1 FSDP module backward-hook path; "
            "PyTorch 2.11 set_all_reduce_hook runs before sharded DTensor.grad assignment"
        )

    def _copy_worker(self) -> None:
        while True:
            task = self._tasks.get()
            task.event.synchronize()
            for buffer, accumulate in task.buffers:
                if accumulate:
                    assert buffer.staging is not None
                    buffer.accumulator.add_(buffer.staging)
                with self._condition:
                    buffer.initialized = True
                    buffer.pending = False
                    self._condition.notify_all()

    def _get_buffer(self, param: nn.Parameter, grad: DTensor) -> _CPUGradientBuffer:
        param_id = id(param)
        if param_id not in self._buffers:
            local_grad = grad.to_local()
            accumulator = torch.empty_like(local_grad, device="cpu", pin_memory=True)
            template = copy.copy(grad)
            template._local_tensor = accumulator
            self._buffers[param_id] = _CPUGradientBuffer(template, accumulator)
        return self._buffers[param_id]

    def _wait_buffer(self, buffer: _CPUGradientBuffer) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not buffer.pending)

    @torch.no_grad()
    def _offload_params(
        self,
        params: list[nn.Parameter],
        *,
        post_reduce_event: torch.cuda.Event | None = None,
    ) -> None:
        copies: list[tuple[_CPUGradientBuffer, bool]] = []
        current_stream = torch.cuda.current_stream()
        self._d2h_stream.wait_stream(current_stream)
        if post_reduce_event is not None:
            self._d2h_stream.wait_event(post_reduce_event)
        with torch.cuda.stream(self._d2h_stream):
            for param in params:
                if id(param) not in self._optimizer_param_ids or param.grad is None:
                    continue
                if not isinstance(param.grad, DTensor):
                    raise TypeError(f"Expected FSDP2 DTensor gradient, got {type(param.grad)}")
                buffer = self._get_buffer(param, param.grad)
                self._wait_buffer(buffer)
                local_grad = param.grad.to_local()
                accumulate = buffer.initialized
                if accumulate and buffer.staging is None:
                    buffer.staging = torch.empty_like(buffer.accumulator, pin_memory=True)
                destination = buffer.staging if accumulate else buffer.accumulator
                assert destination is not None
                destination.copy_(local_grad, non_blocking=True)
                local_grad.record_stream(self._d2h_stream)
                buffer.pending = True
                copies.append((buffer, accumulate))
                param.grad = None
            if copies:
                event = self._d2h_stream.record_event()
                self._tasks.put(_GradientCopyTask(event, copies))

    def _unit(self, module: nn.Module) -> tuple[list[nn.Parameter], object | None]:
        state = module._get_fsdp_state()
        param_group = state._fsdp_param_group
        if param_group is None:
            return [], None
        params = [fsdp_param.sharded_param for fsdp_param in param_group.fsdp_params]
        return params, param_group

    def _backward_hook(self, module: nn.Module, _grad_input, _grad_output) -> None:
        module_id = id(module)
        if module_id in self._seen_units:
            return
        self._seen_units.add(module_id)
        if self._lagged_unit is not None:
            params, param_group = self._lagged_unit
            post_reduce_event = getattr(param_group, "_post_reduce_event", None)
            self._offload_params(params, post_reduce_event=post_reduce_event)
        self._lagged_unit = self._unit(module)

    def finish_backward(self, *, wait_for_copies: bool = True) -> None:
        if self._lagged_unit is not None:
            params, param_group = self._lagged_unit
            post_reduce_event = getattr(param_group, "_post_reduce_event", None)
            self._offload_params(params, post_reduce_event=post_reduce_event)
        remaining = [param for chunk in self._chunks for param in chunk if param.grad is not None]
        self._offload_params(remaining)
        if wait_for_copies:
            self.wait()
        self._lagged_unit = None
        self._seen_units.clear()
        if wait_for_copies:
            allocated = sum(
                buffer.accumulator.nbytes + (buffer.staging.nbytes if buffer.staging is not None else 0)
                for buffer in self._buffers.values()
            )
            if allocated != self._logged_cpu_allocation:
                get_logger().info(f"Gradient CPU offload allocated {allocated / 1024**3:.2f} GiB pinned RAM")
                self._logged_cpu_allocation = allocated

    def wait(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not any(buffer.pending for buffer in self._buffers.values()))

    @torch.no_grad()
    def scale_(self, factor: float) -> None:
        self.wait()
        self._gradient_scale *= factor

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        self.wait()
        local_squared_norm = sum(
            torch.linalg.vector_norm(buffer.accumulator, dtype=torch.float32).square().item()
            for buffer in self._buffers.values()
            if buffer.initialized
        )
        local_squared_norm *= self._gradient_scale**2
        total_norm = torch.tensor(local_squared_norm, dtype=torch.float32, device="cuda")
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
        total_norm.div_(self._dp_replicate).sqrt_()
        clip_coefficient = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0).item()
        self._gradient_scale *= clip_coefficient
        return total_norm

    def prefetch_chunk(self, chunk_idx: int, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            for param in self._chunks[chunk_idx]:
                buffer = self._buffers.get(id(param))
                if buffer is None or not buffer.initialized:
                    param.grad = None
                    continue
                local_grad = buffer.accumulator.to("cuda", non_blocking=True)
                if self._gradient_scale != 1.0:
                    local_grad.mul_(self._gradient_scale)
                grad = copy.copy(buffer.template)
                grad._local_tensor = local_grad
                param.grad = grad

    def release_chunk(self, chunk_idx: int) -> None:
        for param in self._chunks[chunk_idx]:
            param.grad = None

    def zero_grad(self) -> None:
        self.wait()
        for buffer in self._buffers.values():
            buffer.initialized = False
        self._gradient_scale = 1.0


class CPUOffloadOptimizer:
    """Keeps optimizer states on CPU, moving them to GPU only for step().

    Weights stay on GPU (unlike FSDP CPUOffload). With activation checkpointing,
    activations and optimizer states are never on GPU at the same time, so peak
    memory is max(activations, opt_states) instead of sum.

    The step is performed per-transformer-layer with stream-overlapped H2D/D2H
    transfers: while layer *i* computes, layer *i+1* is prefetched and layer *i-1*
    is evicted. The prefetch waits for the eviction to complete, so at most two
    layers' optimizer states are on GPU at any time.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        named_params: list[tuple[str, nn.Parameter]],
        model: nn.Module | None = None,
        grad_cpu_offload: bool = False,
        dp_replicate: int = 1,
        pin_memory: bool = True,
    ):
        self.optimizer = optimizer
        self.pin_memory = pin_memory
        self._initialized = False
        self._chunks = self._build_chunks(named_params)
        # Reuse the transfer streams across steps: fresh streams each step land
        # their H2D/D2H staging in new per-stream allocator pools, growing
        # reserved memory every step and starving the default stream.
        self._h2d_stream = torch.cuda.Stream()
        self._d2h_stream = torch.cuda.Stream()
        if grad_cpu_offload and model is None:
            raise ValueError("Gradient CPU offload requires the model")
        self._gradient_manager = GradientOffloadManager(model, self._chunks, dp_replicate) if grad_cpu_offload else None

    @staticmethod
    def _extract_layer_idx(name: str) -> int | None:
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    def _build_chunks(self, named_params: list[tuple[str, nn.Parameter]]) -> list[list[nn.Parameter]]:
        """Group params by transformer layer; non-layer params go last."""
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
        """Move all optimizer states to *device* (bulk, for state_dict/checkpoint)."""
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, DTensor):
                    new_dtensor = copy.copy(v)
                    new_dtensor._local_tensor = self._move_tensor(v._local_tensor, device)
                    state[k] = new_dtensor
                elif isinstance(v, torch.Tensor):
                    state[k] = self._move_tensor(v, device)

    def _move_chunk_states(self, chunk_idx: int, device: str, stream: torch.cuda.Stream | None = None):
        """Move optimizer states for a single chunk to/from *device*.

        When *stream* is provided the transfers are issued on that stream. For D2H,
        the old GPU tensor is recorded on the stream so the caching allocator won't
        reuse its storage until the async copy finishes.
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
                        new_dtensor = copy.copy(v)
                        new_dtensor._local_tensor = self._move_tensor(v._local_tensor, device)
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
        self._original_param_groups = self.optimizer.param_groups
        original_steps = [g.get("step", 0) for g in self._original_param_groups]

        h2d_stream = self._h2d_stream
        d2h_stream = self._d2h_stream
        compute_stream = torch.cuda.current_stream()
        n = len(self._chunks)

        self._move_chunk_states(0, "cuda", h2d_stream)
        if self._gradient_manager is not None:
            self._gradient_manager.prefetch_chunk(0, h2d_stream)
        for i in range(n):
            compute_stream.wait_stream(h2d_stream)
            if i + 1 < n:
                # Wait for previous D2H before reusing pinned CPU buffers.
                h2d_stream.wait_stream(d2h_stream)
                self._move_chunk_states(i + 1, "cuda", h2d_stream)
                if self._gradient_manager is not None:
                    self._gradient_manager.prefetch_chunk(i + 1, h2d_stream)
            self._step_chunk(i, closure if i == 0 else None)
            if self._gradient_manager is not None:
                self._gradient_manager.release_chunk(i)
            d2h_stream.wait_stream(compute_stream)
            self._move_chunk_states(i, "cpu", d2h_stream)
        torch.cuda.synchronize()

        self._sync_step_counters(original_steps)
        self._original_param_groups = None
        self._initialized = True

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        if self._gradient_manager is not None:
            self._gradient_manager.zero_grad()

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
    grad_cpu_offload: bool = False,
    model: nn.Module | None = None,
) -> tuple[Optimizer | CPUOffloadOptimizer, GradientOffloadManager | None]:
    if grad_cpu_offload and not cpu_offload:
        raise ValueError("Gradient CPU offload requires optimizer CPU offload")
    if lora:
        # Wait for run 0 to be created in the multi run manager
        # Otherwise, the creation will reset the parameters
        multi_run_manager = get_multi_run_manager()
        multi_run_manager.wait_for_run(0)
        named_params = multi_run_manager.get_named_parameters_for_run(0)

    optimizer = _create_optimizer(config, named_params, parallel_dims)

    if cpu_offload:
        offload_description = "optimizer state and gradient" if grad_cpu_offload else "optimizer state"
        get_logger().info(f"Wrapping optimizer for {offload_description} CPU offloading")
        optimizer = CPUOffloadOptimizer(
            optimizer,
            named_params=named_params,
            model=model,
            grad_cpu_offload=grad_cpu_offload,
            dp_replicate=parallel_dims.dp_replicate,
        )
        return optimizer, optimizer._gradient_manager

    return optimizer, None


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
