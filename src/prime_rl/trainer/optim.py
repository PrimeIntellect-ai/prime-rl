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
from prime_rl.trainer.cpu_adam import adamw_step as native_cpu_adamw_step
from prime_rl.trainer.cpu_adam import add_bfloat16_ as native_add_bfloat16_
from prime_rl.trainer.cpu_adam import load_cpu_adamw_kernel
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
    staging_holds_final: bool = False


@dataclass
class _GradientCopyTask:
    event: torch.cuda.Event
    buffers: list[tuple[int, _CPUGradientBuffer, bool, bool]]
    final_backward: bool


@dataclass
class _OptimizerChunkTask:
    chunk_idx: int


@dataclass
class _MasterWeight:
    model_param: nn.Parameter
    cpu_tensor: torch.Tensor
    compute_tensor: torch.Tensor | None = None


class GradientOffloadManager:
    """Offloads finalized FSDP2 sharded gradients from post-accumulate hooks."""

    def __init__(
        self,
        chunks: list[list[nn.Parameter]],
        dp_replicate: int,
        cpu_dtype: torch.dtype | None = None,
        preserve_staging_dtype: bool = False,
        chunk_ready_callback: Callable[[int], None] | None = None,
    ):
        self._chunks = chunks
        self._dp_replicate = dp_replicate
        self._cpu_dtype = cpu_dtype
        self._preserve_staging_dtype = preserve_staging_dtype
        self._optimizer_param_ids = {id(param) for chunk in chunks for param in chunk}
        self._chunk_param_ids = [{id(param) for param in chunk} for chunk in chunks]
        self._chunk_by_param_id = {
            param_id: chunk_idx for chunk_idx, param_ids in enumerate(self._chunk_param_ids) for param_id in param_ids
        }
        self._buffers: dict[int, _CPUGradientBuffer] = {}
        self._chunk_ready_callback = chunk_ready_callback

        self._d2h_stream = torch.cuda.Stream()
        self._tasks: queue.SimpleQueue[_GradientCopyTask | _OptimizerChunkTask | None] = queue.SimpleQueue()
        self._condition = threading.Condition()
        self._gradient_scale = 1.0
        self._final_backward = False
        self._overlap_optimizer = False
        self._final_enqueued_param_ids: set[int] = set()
        self._pending_chunk_param_ids: list[set[int]] = []
        self._scheduled_chunks: set[int] = set()
        self._completed_chunks = 0
        self._worker_error: BaseException | None = None
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
        acc_slab_numel: dict[torch.dtype, int] = {}
        stage_slab_numel: dict[torch.dtype, int] = {}
        offsets: dict[int, tuple[torch.dtype, int, torch.dtype, int, int]] = {}
        for param in dtensor_params:
            local = param.data.to_local()
            acc_dtype = self._cpu_dtype or local.dtype
            stage_dtype = local.dtype if preserve_staging_dtype else acc_dtype
            acc_align = max(1, ALIGN // torch.empty((), dtype=acc_dtype).element_size())
            stage_align = max(1, ALIGN // torch.empty((), dtype=stage_dtype).element_size())
            acc_start = acc_slab_numel.get(acc_dtype, 0)
            stage_start = stage_slab_numel.get(stage_dtype, 0)
            offsets[id(param)] = (acc_dtype, acc_start, stage_dtype, stage_start, local.numel())
            acc_slab_numel[acc_dtype] = acc_start + ((local.numel() + acc_align - 1) // acc_align * acc_align)
            stage_slab_numel[stage_dtype] = stage_start + (
                (local.numel() + stage_align - 1) // stage_align * stage_align
            )
        acc_slabs = {
            dtype: torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
            for dtype, numel in acc_slab_numel.items()
        }
        stage_slabs = {
            dtype: torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
            for dtype, numel in stage_slab_numel.items()
        }
        for param in dtensor_params:
            data = param.data
            local = data.to_local()
            acc_dtype, acc_start, stage_dtype, stage_start, numel = offsets[id(param)]
            accumulator = acc_slabs[acc_dtype].narrow(0, acc_start, numel).view(local.shape)
            staging = stage_slabs[stage_dtype].narrow(0, stage_start, numel).view(local.shape)
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
        try:
            self._copy_worker_loop()
        except BaseException as error:
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()

    def _copy_worker_loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            if isinstance(task, _OptimizerChunkTask):
                self._run_optimizer_chunk(task.chunk_idx)
                continue
            task.event.synchronize()
            for param_id, buffer, accumulate, copied_to_staging in task.buffers:
                assert buffer.staging is not None
                if task.final_backward and not accumulate and copied_to_staging:
                    buffer.staging_holds_final = True
                elif accumulate:
                    if buffer.accumulator.dtype == torch.float32 and buffer.staging.dtype == torch.bfloat16:
                        native_add_bfloat16_(buffer.accumulator, buffer.staging)
                    else:
                        buffer.accumulator.add_(buffer.staging)
                    buffer.staging_holds_final = False
                elif copied_to_staging:
                    buffer.accumulator.copy_(buffer.staging)
                    buffer.staging_holds_final = False
                with self._condition:
                    buffer.initialized = True
                    buffer.pending = False
                    self._condition.notify_all()
                if task.final_backward:
                    self._mark_final_param_ready(param_id)

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("Gradient offload worker failed") from self._worker_error

    def _mark_final_param_ready(self, param_id: int) -> None:
        chunk_idx = self._chunk_by_param_id[param_id]
        should_run = False
        with self._condition:
            pending = self._pending_chunk_param_ids[chunk_idx]
            pending.discard(param_id)
            if not pending and chunk_idx not in self._scheduled_chunks:
                self._scheduled_chunks.add(chunk_idx)
                should_run = True
        if should_run:
            self._run_optimizer_chunk(chunk_idx)

    def _run_optimizer_chunk(self, chunk_idx: int) -> None:
        if self._chunk_ready_callback is None:
            return
        self._chunk_ready_callback(chunk_idx)
        with self._condition:
            self._completed_chunks += 1
            self._condition.notify_all()

    def begin_step(self, gradient_scale: float, *, overlap_optimizer: bool) -> None:
        self.wait()
        with self._condition:
            self._raise_worker_error()
            self._gradient_scale = gradient_scale
            self._overlap_optimizer = overlap_optimizer and self._chunk_ready_callback is not None
            self._final_backward = False
            self._final_enqueued_param_ids.clear()
            self._pending_chunk_param_ids = []
            self._scheduled_chunks.clear()
            self._completed_chunks = 0

    def begin_backward(self, *, final_backward: bool) -> None:
        self._final_backward = final_backward
        if final_backward and self._overlap_optimizer:
            with self._condition:
                self._final_enqueued_param_ids.clear()
                self._pending_chunk_param_ids = [set(param_ids) for param_ids in self._chunk_param_ids]
                self._scheduled_chunks.clear()
                self._completed_chunks = 0

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
            staging = torch.empty_like(
                local_grad,
                dtype=local_grad.dtype if self._preserve_staging_dtype else accumulator.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._buffers[param_id] = _CPUGradientBuffer(template, accumulator, staging)
        return self._buffers[param_id]

    def _wait_buffer(self, buffer: _CPUGradientBuffer) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._worker_error is not None or not buffer.pending)
            self._raise_worker_error()

    @torch.no_grad()
    def _offload_params(
        self,
        params: list[nn.Parameter],
    ) -> None:
        copies: list[tuple[int, _CPUGradientBuffer, bool, bool]] = []
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
                if buffer.staging is None:
                    buffer.staging = torch.empty_like(buffer.accumulator, pin_memory=True)
                copied_to_staging = accumulate or (
                    self._preserve_staging_dtype
                    and self._final_backward
                    and self._overlap_optimizer
                    and buffer.staging.dtype != buffer.accumulator.dtype
                )
                destination = buffer.staging if copied_to_staging else buffer.accumulator
                assert destination is not None
                destination.copy_(local_grad, non_blocking=True)
                local_grad.record_stream(self._d2h_stream)
                buffer.pending = True
                param_id = id(param)
                copies.append((param_id, buffer, accumulate, copied_to_staging))
                if self._final_backward and self._overlap_optimizer:
                    self._final_enqueued_param_ids.add(param_id)
                param.grad = None
            if copies:
                event = self._d2h_stream.record_event()
                self._tasks.put(_GradientCopyTask(event, copies, self._final_backward and self._overlap_optimizer))

    def _offload_hook(self, param: torch.Tensor) -> None:
        if not isinstance(param, nn.Parameter):
            raise TypeError(f"Expected parameter in post-accumulate hook, got {type(param)}")
        self._offload_params([param])

    def finish_backward(self, *, wait_for_copies: bool = True) -> None:
        remaining = [param for chunk in self._chunks for param in chunk if param.grad is not None]
        self._offload_params(remaining)
        if self._final_backward and self._overlap_optimizer:
            missing_param_ids = self._optimizer_param_ids - self._final_enqueued_param_ids
            ready_chunks = []
            with self._condition:
                for param_id in missing_param_ids:
                    chunk_idx = self._chunk_by_param_id[param_id]
                    pending = self._pending_chunk_param_ids[chunk_idx]
                    pending.discard(param_id)
                    if not pending and chunk_idx not in self._scheduled_chunks:
                        self._scheduled_chunks.add(chunk_idx)
                        ready_chunks.append(chunk_idx)
            for chunk_idx in ready_chunks:
                self._tasks.put(_OptimizerChunkTask(chunk_idx))
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
            self._condition.wait_for(
                lambda: self._worker_error is not None or not any(buffer.pending for buffer in self._buffers.values())
            )
            self._raise_worker_error()

    def wait_for_optimizer(self) -> None:
        if not self._overlap_optimizer:
            return
        with self._condition:
            self._condition.wait_for(
                lambda: self._worker_error is not None or self._completed_chunks == len(self._chunks)
            )
            self._raise_worker_error()

    @property
    def optimizer_overlapped(self) -> bool:
        return self._overlap_optimizer

    @property
    def gradient_scale(self) -> float:
        return self._gradient_scale

    @torch.no_grad()
    def scale_(self, factor: float) -> None:
        self.wait()
        self._gradient_scale *= factor

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        if self._overlap_optimizer:
            raise RuntimeError("Gradient clipping cannot run after optimizer-in-backward has started")
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

    def load_cpu_chunk(
        self,
        chunk_idx: int,
        target_params: list[nn.Parameter],
        *,
        wait: bool = True,
        apply_scale: bool = True,
        allow_staging_gradient: bool = False,
        assign_grad: bool = True,
    ) -> list[torch.Tensor | None]:
        if wait:
            self.wait()
        source_params = self._chunks[chunk_idx]
        if len(source_params) != len(target_params):
            raise ValueError("Gradient source and target chunks must have the same size")
        gradients: list[torch.Tensor | None] = []
        for source_param, target_param in zip(source_params, target_params):
            buffer = self._buffers.get(id(source_param))
            if buffer is None or not buffer.initialized:
                gradients.append(None)
                if assign_grad:
                    target_param.grad = None
                continue
            grad = (
                buffer.staging
                if allow_staging_gradient and buffer.staging_holds_final and buffer.staging is not None
                else buffer.accumulator.float()
            )
            if apply_scale and self._gradient_scale != 1.0:
                grad.mul_(self._gradient_scale)
            gradients.append(grad)
            if assign_grad:
                target_param.grad = grad
        return gradients

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
            buffer.staging_holds_final = False
        self._gradient_scale = 1.0

    def close(self) -> None:
        if self._closed:
            return
        self.wait_for_optimizer()
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

    _TARGET_GPU_CHUNK_NUMEL = 128 * 1024**2
    _TARGET_CPU_CHUNK_NUMEL = 16 * 1024**2
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
        self._master_weights = master_weights
        self._chunks = self._build_chunks()
        # Reuse the transfer streams across steps: fresh streams each step land
        # their H2D/D2H staging in new per-stream allocator pools, growing
        # reserved memory every step and starving the default stream.
        self._h2d_stream = torch.cuda.Stream()
        self._d2h_stream = torch.cuda.Stream()
        self._cuda_device = torch.cuda.current_device()
        self._all_param_groups = self.optimizer.param_groups
        self._chunk_groups = self._build_chunk_groups()
        self._native_cpu_adamw = (
            master_weights is not None
            and isinstance(optimizer, AdamW)
            and offload_config.cpu_optimizer_backend == "native"
        )
        self._fused_cpu_adamw = (
            master_weights is not None
            and isinstance(optimizer, AdamW)
            and all(group["fused"] for group in optimizer.param_groups)
            and not self._native_cpu_adamw
        )
        self._adamw_grad_scale = torch.ones((), dtype=torch.float32) if self._fused_cpu_adamw else None
        self._adamw_found_inf = torch.zeros((), dtype=torch.float32) if self._fused_cpu_adamw else None
        if self._native_cpu_adamw:
            get_logger().info("Loading native read-only-gradient multi-tensor CPU AdamW kernel")
            load_cpu_adamw_kernel()
        gradient_chunks = self._chunks
        if master_weights is not None:
            gradient_chunks = [[master_weights[id(param)].model_param for param in chunk] for chunk in self._chunks]
        self._gradient_manager = (
            GradientOffloadManager(
                gradient_chunks,
                dp_replicate,
                cpu_dtype=torch.float32 if master_weights is not None else None,
                preserve_staging_dtype=self._native_cpu_adamw,
                chunk_ready_callback=self._step_cpu_chunk if master_weights is not None else None,
            )
            if offload_config.gradients or offload_config.full
            else None
        )
        if master_weights is not None:
            self._initialize_cpu_optimizer_state()

    def _build_chunks(self) -> list[list[nn.Parameter]]:
        """Group optimizer parameters into bounded, model-independent chunks."""
        target_numel = (
            self._TARGET_CPU_CHUNK_NUMEL if self._master_weights is not None else self._TARGET_GPU_CHUNK_NUMEL
        )
        chunks: list[list[nn.Parameter]] = []
        chunk: list[nn.Parameter] = []
        chunk_numel = 0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if chunk and chunk_numel + param.numel() > target_numel:
                    chunks.append(chunk)
                    chunk = []
                    chunk_numel = 0
                chunk.append(param)
                chunk_numel += param.numel()
        if chunk:
            chunks.append(chunk)
        return chunks

    def _build_chunk_groups(self) -> list[list[tuple[dict, list[nn.Parameter]]]]:
        chunk_by_param_id = {id(param): chunk_idx for chunk_idx, chunk in enumerate(self._chunks) for param in chunk}
        chunk_groups: list[list[tuple[dict, list[nn.Parameter]]]] = [[] for _ in self._chunks]
        for group in self.optimizer.param_groups:
            params_by_chunk: dict[int, list[nn.Parameter]] = defaultdict(list)
            for param in group["params"]:
                params_by_chunk[chunk_by_param_id[id(param)]].append(param)
            for chunk_idx, params in params_by_chunk.items():
                chunk_groups[chunk_idx].append((group, params))
        return chunk_groups

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
                source = master.compute_tensor if master.compute_tensor is not None else master.cpu_tensor
                master.model_param.to_local().copy_(source, non_blocking=True)

    def _step_chunk(self, chunk_idx: int, closure=None):
        """Run optimizer.step() for a single chunk by temporarily swapping param_groups."""
        chunk_groups = []
        for group, params in self._chunk_groups[chunk_idx]:
            new_group = {k: v for k, v in group.items() if k != "params"}
            new_group["params"] = params
            chunk_groups.append(new_group)
        self.optimizer.param_groups = chunk_groups
        result = self.optimizer.step(closure)
        self.optimizer.param_groups = self._all_param_groups
        return result

    def _step_native_cpu_adamw_chunk(
        self,
        chunk_idx: int,
        gradients: list[torch.Tensor | None],
    ) -> None:
        assert self._gradient_manager is not None
        assert self._master_weights is not None
        gradient_by_param_id = {
            id(param): gradient for param, gradient in zip(self._chunks[chunk_idx], gradients) if gradient is not None
        }
        for group, params in self._chunk_groups[chunk_idx]:
            params_with_grad = [param for param in params if id(param) in gradient_by_param_id]
            if not params_with_grad:
                continue
            states = [self.optimizer.state[param] for param in params_with_grad]
            compute_params = []
            for param in params_with_grad:
                compute_param = self._master_weights[id(param)].compute_tensor
                if compute_param is None:
                    raise RuntimeError("Native CPU AdamW requires BF16 compute-weight shadows")
                compute_params.append(compute_param)
            beta1, beta2 = group["betas"]
            native_cpu_adamw_step(
                params_with_grad,
                [gradient_by_param_id[id(param)] for param in params_with_grad],
                [state["exp_avg"] for state in states],
                [state["exp_avg_sq"] for state in states],
                [state["step"] for state in states],
                compute_params,
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                gradient_scale=self._gradient_manager.gradient_scale,
            )

    def _step_cpu_chunk(self, chunk_idx: int) -> None:
        assert self._gradient_manager is not None
        self._prepare_fused_cpu_adamw()
        gradients = self._gradient_manager.load_cpu_chunk(
            chunk_idx,
            self._chunks[chunk_idx],
            wait=False,
            apply_scale=not (self._native_cpu_adamw or self._fused_cpu_adamw),
            allow_staging_gradient=self._native_cpu_adamw,
            assign_grad=not self._native_cpu_adamw,
        )
        if self._native_cpu_adamw:
            self._step_native_cpu_adamw_chunk(chunk_idx, gradients)
        else:
            self._step_chunk(chunk_idx)
        self._gradient_manager.release_chunk(chunk_idx, self._chunks[chunk_idx])
        with torch.cuda.device(self._cuda_device):
            self._update_compute_weights(chunk_idx, self._h2d_stream)

    def _prepare_fused_cpu_adamw(self) -> None:
        if not self._fused_cpu_adamw:
            return
        assert self._gradient_manager is not None
        assert self._adamw_grad_scale is not None
        assert self._adamw_found_inf is not None
        self._adamw_grad_scale.fill_(1.0 / self._gradient_manager.gradient_scale)
        self.optimizer.grad_scale = self._adamw_grad_scale
        self.optimizer.found_inf = self._adamw_found_inf

    def _record_native_optimizer_step(self) -> None:
        if self._native_cpu_adamw:
            # LRScheduler normally sets this marker through its optimizer.step wrapper.
            self.optimizer._opt_called = True

    def _sync_step_counters(self, original_steps: list):
        """Increment the original param_groups' step counters by one.

        Muon stores a per-group ``step`` that ``step()`` increments. Because we swap
        in per-chunk copies, the original groups never see the increment. Standard
        optimizers (AdamW, SGD, SignSGD) keep step in ``state[p]['step']`` which is
        per-parameter and already correct.
        """
        for group, orig_step in zip(self._all_param_groups, original_steps):
            group["step"] = orig_step + 1

    def step(self, closure=None):
        if self._master_weights is not None:
            assert self._gradient_manager is not None
            if self._gradient_manager.optimizer_overlapped:
                if closure is not None:
                    raise ValueError("Optimizer closures are not supported with optimizer-in-backward")
                self._gradient_manager.wait_for_optimizer()
                torch.cuda.current_stream().wait_stream(self._h2d_stream)
                self._record_native_optimizer_step()
                self._initialized = True
                return
            self._prepare_fused_cpu_adamw()
            gradients_by_chunk = []
            for i in range(len(self._chunks)):
                gradients_by_chunk.append(
                    self._gradient_manager.load_cpu_chunk(
                        i,
                        self._chunks[i],
                        apply_scale=not (self._native_cpu_adamw or self._fused_cpu_adamw),
                        allow_staging_gradient=self._native_cpu_adamw,
                        assign_grad=not self._native_cpu_adamw,
                    )
                )
            if self._native_cpu_adamw:
                if closure is not None:
                    raise ValueError("Optimizer closures are not supported with native CPU AdamW")
                for i in range(len(self._chunks)):
                    self._step_native_cpu_adamw_chunk(i, gradients_by_chunk[i])
            else:
                self.optimizer.step(closure)
            for i in range(len(self._chunks)):
                self._gradient_manager.release_chunk(i, self._chunks[i])
                self._update_compute_weights(i, self._h2d_stream)
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
            torch.cuda.synchronize()
            self._record_native_optimizer_step()
            self._initialized = True
            return

        original_steps = [g.get("step", 0) for g in self._all_param_groups]
        if self._gradient_manager is not None:
            self._gradient_manager.wait()
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
        if self._native_cpu_adamw:
            self._initialize_native_cpu_adamw_state()
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
        for state in self.optimizer.state.values():
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                step.zero_()
            elif step is not None:
                state["step"] = 0
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
            if "step" in group:
                group["step"] = 0

    def _initialize_native_cpu_adamw_state(self) -> None:
        alignment = 256 // torch.empty((), dtype=torch.float32).element_size()
        offsets = {}
        slab_numel = 0
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                offsets[id(param)] = slab_numel
                slab_numel += (param.numel() + alignment - 1) // alignment * alignment
        exp_avg_slab = torch.zeros(slab_numel, dtype=torch.float32, device="cpu")
        exp_avg_sq_slab = torch.zeros_like(exp_avg_slab)
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                offset = offsets[id(param)]
                self.optimizer.state[param] = {
                    "step": torch.zeros((), dtype=torch.float32, device="cpu"),
                    "exp_avg": exp_avg_slab.narrow(0, offset, param.numel()).view_as(param),
                    "exp_avg_sq": exp_avg_sq_slab.narrow(0, offset, param.numel()).view_as(param),
                }

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
            for master in self._master_weights.values():
                if master.compute_tensor is not None:
                    master.compute_tensor.copy_(master.cpu_tensor)
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
    if offload_config is not None and offload_config.full and config.max_norm is not None:
        get_logger().warning("Disabling gradient clipping because full optimizer CPU offload updates during backward")
        config.max_norm = None
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
            create_compute_shadow=(config.type == "adamw" and offload_config.cpu_optimizer_backend == "native"),
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
    *,
    create_compute_shadow: bool = False,
) -> tuple[list[tuple[str, nn.Parameter]], dict[int, _MasterWeight]]:
    trainable_params = [(name, model_param) for name, model_param in named_params if model_param.requires_grad]
    if not trainable_params:
        raise ValueError("Gradient CPU offload requires trainable parameters")
    alignment_dtype = torch.bfloat16 if create_compute_shadow else torch.float32
    alignment = 256 // torch.empty((), dtype=alignment_dtype).element_size()
    offsets = {}
    slab_numel = 0
    for name, model_param in trainable_params:
        if not isinstance(model_param, DTensor):
            raise TypeError(f"Expected FSDP2 DTensor parameter, got {type(model_param)}")
        offsets[name] = slab_numel
        local_numel = model_param.to_local().numel()
        slab_numel += (local_numel + alignment - 1) // alignment * alignment
    master_slab = torch.empty(slab_numel, dtype=torch.float32, device="cpu", pin_memory=True)
    compute_slab = (
        torch.empty(slab_numel, dtype=torch.bfloat16, device="cpu", pin_memory=True) if create_compute_shadow else None
    )
    master_named_params = []
    master_weights = {}
    for name, model_param in trainable_params:
        local_param = model_param.to_local()
        cpu_tensor = master_slab.narrow(0, offsets[name], local_param.numel()).view(local_param.shape)
        cpu_tensor.copy_(local_param, non_blocking=True)
        compute_tensor = (
            compute_slab.narrow(0, offsets[name], local_param.numel()).view(local_param.shape)
            if compute_slab is not None
            else None
        )
        if compute_tensor is not None:
            compute_tensor.copy_(local_param, non_blocking=True)
        master_param = nn.Parameter(cpu_tensor, requires_grad=True)
        master_named_params.append((name, master_param))
        master_weights[id(master_param)] = _MasterWeight(model_param, cpu_tensor, compute_tensor)
    torch.cuda.synchronize()

    original_buffers = {name: buffer.detach() for name, buffer in model.named_buffers() if buffer.is_floating_point()}
    model.to(dtype=torch.bfloat16)
    for name, buffer in model.named_buffers():
        if name in original_buffers:
            buffer.data = original_buffers[name]
    del local_param
    torch.cuda.empty_cache()

    master_allocation = sum(master.cpu_tensor.nbytes for master in master_weights.values())
    shadow_allocation = sum(
        master.compute_tensor.nbytes for master in master_weights.values() if master.compute_tensor is not None
    )
    get_logger().info(
        f"CPU optimizer step allocated {master_allocation / 1024**3:.2f} GiB of FP32 masters"
        f"{f' and {shadow_allocation / 1024**3:.2f} GiB of BF16 compute shadows' if shadow_allocation else ''} "
        "in pinned RAM; persistent GPU parameters use BF16"
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
