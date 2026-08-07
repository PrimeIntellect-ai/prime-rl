import copy
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
from dion import Muon
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import SGD, AdamW, Optimizer

from prime_rl.configs.trainer import OptimizerConfig, OptimizerOffloadingConfig
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


@dataclass
class _MasterWeight:
    model_param: nn.Parameter
    cpu_tensor: torch.Tensor


class GradientOffloadManager:
    """Offloads finalized FSDP2 sharded gradients from post-accumulate hooks."""

    def __init__(
        self,
        chunks: list[list[nn.Parameter]],
        dp_replicate: int,
        cpu_dtype: torch.dtype | None = None,
    ):
        self._chunks = chunks
        self._dp_replicate = dp_replicate
        self._cpu_dtype = cpu_dtype
        self._optimizer_param_ids = {id(param) for chunk in chunks for param in chunk}
        self._buffers: dict[int, _CPUGradientBuffer] = {}

        self._d2h_stream = torch.cuda.Stream()
        self._tasks: queue.SimpleQueue[_GradientCopyTask | None] = queue.SimpleQueue()
        self._condition = threading.Condition()
        self._gradient_scale = 1.0
        self._logged_cpu_allocation = 0
        self._closed = False

        params = {id(param): param for chunk in chunks for param in chunk}
        # Pin every buffer NOW, before the first collective of the job. The
        # post-accumulate hook runs inside FSDP's foreach_reduce on the autograd
        # thread; a lazy cudaHostAlloc there stalls against the device while
        # peer ranks' NCCL kernels spin waiting for this rank's next collective
        # — a distributed deadlock (deterministic at 78 layers, seen from 30k to
        # 131k tokens). Init-time pinning is the only safe window.
        #
        # Slab-allocate: one pinned block per dtype for accumulators and one for
        # staging, carved into per-param views. ~800 per-param cudaHostAllocs
        # (~1.4 TiB/node) took 10-20 min with large node skew; a handful of huge
        # allocations pins the same bytes in well under a minute.
        dtensor_params = [p for p in params.values() if isinstance(p.data, DTensor)]
        ALIGN = 256  # bytes; keeps carved views DMA-friendly
        slab_numel: dict[torch.dtype, int] = {}
        offsets: dict[int, tuple[torch.dtype, int, int]] = {}
        for param in dtensor_params:
            local = param.data.to_local()
            dtype = self._cpu_dtype or local.dtype
            align_elems = max(1, ALIGN // local.element_size())
            start = slab_numel.get(dtype, 0)
            offsets[id(param)] = (dtype, start, local.numel())
            padded = (local.numel() + align_elems - 1) // align_elems * align_elems
            slab_numel[dtype] = start + padded
        acc_slabs = {dt: torch.empty(n, dtype=dt, device="cpu", pin_memory=True) for dt, n in slab_numel.items()}
        stage_slabs = {dt: torch.empty(n, dtype=dt, device="cpu", pin_memory=True) for dt, n in slab_numel.items()}
        for param in dtensor_params:
            data = param.data
            local = data.to_local()
            dtype, start, numel = offsets[id(param)]
            accumulator = acc_slabs[dtype].narrow(0, start, numel).view(local.shape)
            staging = stage_slabs[dtype].narrow(0, start, numel).view(local.shape)
            template = copy.copy(data.detach())
            template._local_tensor = accumulator
            self._buffers[id(param)] = _CPUGradientBuffer(template, accumulator, staging=staging)
        preallocated = sum(s.nbytes for s in acc_slabs.values()) + sum(s.nbytes for s in stage_slabs.values())
        self._hook_handles = [param.register_post_accumulate_grad_hook(self._offload_hook) for param in params.values()]
        self._worker = threading.Thread(target=self._copy_worker, name="grad-offload", daemon=True)
        self._worker.start()
        get_logger().info(
            "Gradient CPU offload uses FSDP sharded-parameter post-accumulate hooks "
            f"({preallocated / 1024**3:.2f} GiB pinned accumulator+staging preallocated)"
        )

    def _copy_worker(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
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
            accumulator = torch.empty_like(
                local_grad,
                dtype=self._cpu_dtype or local_grad.dtype,
                device="cpu",
                pin_memory=True,
            )
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
    ) -> None:
        copies: list[tuple[_CPUGradientBuffer, bool]] = []
        current_stream = torch.cuda.current_stream()
        self._d2h_stream.wait_stream(current_stream)
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

    def _offload_hook(self, param: torch.Tensor) -> None:
        if not isinstance(param, nn.Parameter):
            raise TypeError(f"Expected parameter in post-accumulate hook, got {type(param)}")
        self._offload_params([param])

    def finish_backward(self, *, wait_for_copies: bool = True) -> None:
        remaining = [param for chunk in self._chunks for param in chunk if param.grad is not None]
        self._offload_params(remaining)
        if wait_for_copies:
            self.wait()
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

    def prefetch_chunk(
        self,
        chunk_idx: int,
        stream: torch.cuda.Stream,
        target_params: list[nn.Parameter] | None = None,
    ) -> None:
        source_params = self._chunks[chunk_idx]
        if target_params is None:
            target_params = source_params
        if len(source_params) != len(target_params):
            raise ValueError("Gradient source and target chunks must have the same size")
        with torch.cuda.stream(stream):
            for source_param, target_param in zip(source_params, target_params):
                buffer = self._buffers.get(id(source_param))
                if buffer is None or not buffer.initialized:
                    target_param.grad = None
                    continue
                local_grad = buffer.accumulator.to("cuda", non_blocking=True)
                if self._gradient_scale != 1.0:
                    local_grad.mul_(self._gradient_scale)
                if target_param is source_param:
                    grad = copy.copy(buffer.template)
                    grad._local_tensor = local_grad
                    target_param.grad = grad
                else:
                    target_param.grad = local_grad.to(dtype=target_param.dtype)

    def record_chunk_gradients(self, chunk_idx: int, stream: torch.cuda.Stream) -> None:
        """Keep H2D gradient allocations alive until GPU optimizer kernels finish."""
        for param in self._chunks[chunk_idx]:
            grad = param.grad
            if isinstance(grad, DTensor):
                grad.to_local().record_stream(stream)
            elif isinstance(grad, torch.Tensor):
                grad.record_stream(stream)

    def load_cpu_chunk(self, chunk_idx: int, target_params: list[nn.Parameter]) -> None:
        self.wait()
        source_params = self._chunks[chunk_idx]
        if len(source_params) != len(target_params):
            raise ValueError("Gradient source and target chunks must have the same size")
        for source_param, target_param in zip(source_params, target_params):
            buffer = self._buffers.get(id(source_param))
            if buffer is None or not buffer.initialized:
                target_param.grad = None
                continue
            grad = buffer.accumulator.float()
            if self._gradient_scale != 1.0:
                grad.mul_(self._gradient_scale)
            target_param.grad = grad

    def release_chunk(self, chunk_idx: int, target_params: list[nn.Parameter] | None = None) -> None:
        for param in self._chunks[chunk_idx]:
            param.grad = None
        if target_params is not None:
            for param in target_params:
                param.grad = None

    def zero_grad(self) -> None:
        self.wait()
        for buffer in self._buffers.values():
            buffer.initialized = False
        self._gradient_scale = 1.0

    def close(self) -> None:
        if self._closed:
            return
        self.wait()
        for handle in self._hook_handles:
            handle.remove()
        self._tasks.put(None)
        self._worker.join()
        self._closed = True


class CPUOffloadOptimizer:
    """Keeps optimizer states on CPU, moving them to GPU only for step().

    Weights stay on GPU (unlike FSDP CPUOffload). With activation checkpointing,
    activations and optimizer states are never on GPU at the same time, so peak
    memory is max(activations, opt_states) instead of sum.

    A GPU step is performed in bounded chunks with stream-overlapped H2D/D2H
    transfers: while chunk *i* computes, chunk *i+1* is prefetched and chunk *i-1*
    is evicted.

    With gradient offload, the optimizer instead runs on CPU-resident FP32 masters
    and only refreshed BF16 compute weights are copied to GPU.
    """

    _TARGET_CHUNK_NUMEL = 128 * 1024**2
    _MASTER_WEIGHT_STATE = "prime_rl_master_weight"

    def __init__(
        self,
        optimizer: Optimizer,
        offload_config: OptimizerOffloadingConfig,
        master_weights: dict[int, _MasterWeight] | None = None,
        dp_replicate: int = 1,
    ):
        self.optimizer = optimizer
        self.offload_config = offload_config
        self._initialized = False
        self._chunks = self._build_chunks()
        # Reuse the transfer streams across steps: fresh streams each step land
        # their H2D/D2H staging in new per-stream allocator pools, growing
        # reserved memory every step and starving the default stream.
        self._h2d_stream = torch.cuda.Stream()
        self._d2h_stream = torch.cuda.Stream()
        self._master_weights = master_weights
        gradient_chunks = self._chunks
        if master_weights is not None:
            gradient_chunks = [[master_weights[id(param)].model_param for param in chunk] for chunk in self._chunks]
        self._gradient_manager = (
            GradientOffloadManager(
                gradient_chunks,
                dp_replicate,
                cpu_dtype=torch.float32 if master_weights is not None else None,
            )
            if offload_config.gradients or offload_config.full
            else None
        )

    def _build_chunks(self) -> list[list[nn.Parameter]]:
        """Group optimizer parameters into bounded, model-independent chunks."""
        chunks: list[list[nn.Parameter]] = []
        chunk: list[nn.Parameter] = []
        chunk_numel = 0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if chunk and chunk_numel + param.numel() > self._TARGET_CHUNK_NUMEL:
                    chunks.append(chunk)
                    chunk = []
                    chunk_numel = 0
                chunk.append(param)
                chunk_numel += param.numel()
        if chunk:
            chunks.append(chunk)
        return chunks

    def _move_tensor(self, v: torch.Tensor, device: str) -> torch.Tensor:
        if device == "cpu":
            dst = torch.empty_like(v, device="cpu", pin_memory=True)
            dst.copy_(v, non_blocking=True)
            return dst
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

    @torch.no_grad()
    def _update_compute_weights(self, chunk_idx: int, stream: torch.cuda.Stream) -> None:
        assert self._master_weights is not None
        with torch.cuda.stream(stream):
            for param in self._chunks[chunk_idx]:
                master = self._master_weights[id(param)]
                if not isinstance(master.model_param, DTensor):
                    raise TypeError(f"Expected FSDP2 DTensor parameter, got {type(master.model_param)}")
                master.model_param.to_local().copy_(master.cpu_tensor, non_blocking=True)

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
        if self._master_weights is not None:
            assert self._gradient_manager is not None
            for i in range(len(self._chunks)):
                self._gradient_manager.load_cpu_chunk(i, self._chunks[i])
            self.optimizer.step(closure)
            for i in range(len(self._chunks)):
                self._gradient_manager.release_chunk(i, self._chunks[i])
                self._update_compute_weights(i, self._h2d_stream)
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
            torch.cuda.synchronize()
            self._initialized = True
            return

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
            if self._gradient_manager is not None:
                self._gradient_manager.record_chunk_gradients(i, compute_stream)
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

    @property
    def steps_on_cpu(self) -> bool:
        return self._master_weights is not None

    @torch.no_grad()
    def _initialize_cpu_optimizer_state(self) -> None:
        if self.optimizer.state:
            return
        lrs = []
        for group in self.optimizer.param_groups:
            lrs.append(group["lr"])
            group["lr"] = 0.0
            for param in group["params"]:
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    @staticmethod
    def _checkpoint_dtensor(local_tensor: torch.Tensor, model_param: nn.Parameter) -> DTensor:
        if not isinstance(model_param, DTensor):
            raise TypeError(f"Expected FSDP2 DTensor parameter, got {type(model_param)}")
        # Preserve the model parameter's global shape and placements while DCP
        # reads or writes the optimizer-owned shard directly in CPU memory.
        return DTensor(local_tensor, model_param._spec, requires_grad=False)

    @torch.no_grad()
    def checkpoint_optimizer(self) -> Optimizer:
        """Expose CPU masters and optimizer state under their model parameter FQNs."""
        if self._master_weights is None:
            return self.optimizer
        self._initialize_cpu_optimizer_state()

        checkpoint_optimizer = copy.copy(self.optimizer)
        checkpoint_optimizer.param_groups = []
        checkpoint_optimizer.state = defaultdict(dict)
        for group in self.optimizer.param_groups:
            checkpoint_group = {key: value for key, value in group.items() if key != "params"}
            checkpoint_params = []
            for cpu_param in group["params"]:
                master = self._master_weights[id(cpu_param)]
                model_param = master.model_param
                checkpoint_params.append(model_param)
                checkpoint_state = {}
                for key, value in self.optimizer.state[cpu_param].items():
                    checkpoint_state[key] = (
                        self._checkpoint_dtensor(value, model_param)
                        if isinstance(value, torch.Tensor) and value.shape == cpu_param.shape
                        else value
                    )
                checkpoint_state[self._MASTER_WEIGHT_STATE] = self._checkpoint_dtensor(master.cpu_tensor, model_param)
                checkpoint_optimizer.state[model_param] = checkpoint_state
            checkpoint_group["params"] = checkpoint_params
            checkpoint_optimizer.param_groups.append(checkpoint_group)
        return checkpoint_optimizer

    @torch.no_grad()
    def finish_checkpoint_load(self) -> None:
        if self._master_weights is not None:
            for i in range(len(self._chunks)):
                self._update_compute_weights(i, self._h2d_stream)
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
            torch.cuda.synchronize()
        self._initialized = True


def setup_optimizer(
    config: OptimizerConfig,
    named_params: list[tuple[str, nn.Parameter]],
    parallel_dims: ParallelDims,
    lora: bool = False,
    offload_config: OptimizerOffloadingConfig | None = None,
    model: nn.Module | None = None,
) -> tuple[Optimizer | CPUOffloadOptimizer, GradientOffloadManager | None]:
    if offload_config is not None and offload_config.full and config.type == "muon":
        raise ValueError("Full optimizer CPU offload does not support Muon because the optimizer step runs on CPU")
    if lora:
        # Wait for run 0 to be created in the multi run manager
        # Otherwise, the creation will reset the parameters
        multi_run_manager = get_multi_run_manager()
        multi_run_manager.wait_for_run(0)
        named_params = multi_run_manager.get_named_parameters_for_run(0)

    master_weights = None
    optimizer_named_params = named_params
    if offload_config is not None and offload_config.full:
        if model is None:
            raise ValueError("Gradient CPU offload requires the model")
        optimizer_named_params, master_weights = _create_cpu_master_weights(
            model,
            named_params,
        )

    optimizer = _create_optimizer(
        config,
        optimizer_named_params,
        parallel_dims,
        fused_adamw=offload_config is not None and offload_config.full and config.type == "adamw",
    )

    if offload_config is not None:
        offload_parts = ["optimizer state"]
        if offload_config.full:
            offload_parts.extend(("gradient", "CPU optimizer step"))
        elif offload_config.gradients:
            offload_parts.append("gradient")
        offload_description = ", ".join(offload_parts)
        get_logger().info(f"Using CPU offload for {offload_description}")
        optimizer = CPUOffloadOptimizer(
            optimizer,
            offload_config=offload_config,
            master_weights=master_weights,
            dp_replicate=parallel_dims.dp_replicate,
        )
        return optimizer, optimizer._gradient_manager

    return optimizer, None


@torch.no_grad()
def _create_cpu_master_weights(
    model: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
) -> tuple[list[tuple[str, nn.Parameter]], dict[int, _MasterWeight]]:
    master_named_params = []
    master_weights = {}
    for name, model_param in named_params:
        if not model_param.requires_grad:
            continue
        if not isinstance(model_param, DTensor):
            raise TypeError(f"Expected FSDP2 DTensor parameter, got {type(model_param)}")
        local_param = model_param.to_local()
        cpu_tensor = torch.empty_like(local_param, dtype=torch.float32, device="cpu", pin_memory=True)
        cpu_tensor.copy_(local_param, non_blocking=True)
        master_param = nn.Parameter(cpu_tensor, requires_grad=True)
        master_named_params.append((name, master_param))
        master_weights[id(master_param)] = _MasterWeight(model_param, cpu_tensor)
    if not master_named_params:
        raise ValueError("Gradient CPU offload requires trainable parameters")
    torch.cuda.synchronize()

    original_buffers = {name: buffer.detach() for name, buffer in model.named_buffers() if buffer.is_floating_point()}
    model.to(dtype=torch.bfloat16)
    for name, buffer in model.named_buffers():
        if name in original_buffers:
            buffer.data = original_buffers[name]
    del local_param
    torch.cuda.empty_cache()

    allocation = sum(master.cpu_tensor.nbytes for master in master_weights.values())
    get_logger().info(
        f"CPU optimizer step allocated {allocation / 1024**3:.2f} GiB of FP32 masters in pinned RAM; "
        "persistent GPU parameters use BF16"
    )
    return master_named_params, master_weights


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
