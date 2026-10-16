# This file is fully AI-generated.

import copy
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.distributed as dist
from dion import Muon
from dion.muon import muon_update_batch_async
from dion.opt_utils import create_param_batches, pad_batch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import AdamW, Optimizer

from prime_rl.trainer.optim.base import OffloadOptimizer
from prime_rl.trainer.optim.cpu_adam import adamw_step as native_cpu_adamw_step
from prime_rl.trainer.optim.cpu_adam import add_bfloat16_ as native_add_bfloat16_
from prime_rl.trainer.optim.cpu_adam import copy_or_add_bfloat16_multi_ as native_copy_or_add_bfloat16_multi_
from prime_rl.trainer.optim.cpu_adam import load_cpu_adamw_kernel
from prime_rl.trainer.optim.cpu_adam import sign_sgd_step as native_cpu_sign_sgd_step
from prime_rl.trainer.optim.sign_sgd import SignSGD
from prime_rl.utils.logger import get_logger


@dataclass
class _CPUGradientBuffer:
    template: DTensor
    accumulator: torch.Tensor
    initialized: bool = False
    pending: bool = False
    pending_step: int = -1
    pending_backward: int = -1
    last_enqueued_backward: int = -1
    pending_generations: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class _PinnedTransferSlot:
    index: int
    tensor: torch.Tensor
    event: torch.cuda.Event


@dataclass
class _GradientTransferRequest:
    param_id: int
    local_grad: torch.Tensor
    ready_event: torch.cuda.Event
    step: int
    backward: int
    final_backward: bool


@dataclass
class _GradientCopyTask:
    request: _GradientTransferRequest
    slot: _PinnedTransferSlot


@dataclass
class _BackwardBoundaryTask:
    step: int
    backward: int
    final_backward: bool
    missing_param_ids: set[int]


@dataclass
class _MasterWeight:
    model_param: nn.Parameter
    cpu_tensor: torch.Tensor
    compute_dtype: torch.dtype
    gradient_dtype: torch.dtype


class GradientOffloadManager:
    """Full-offload gradient pipeline with pageable FP32 accumulators and bounded pinned transfer buffers."""

    def __init__(
        self,
        chunks: list[list[nn.Parameter]],
        dp_replicate: int,
        buffer_count: int,
        max_inflight_backwards: int,
        timeout_seconds: float,
        target_chunk_numel: int,
        chunk_ready_callback: Callable[[int], None],
        gradient_dtypes: dict[int, torch.dtype],
        compute_dtypes: dict[int, torch.dtype],
    ):
        self._chunks = chunks
        self._dp_replicate = dp_replicate
        self._chunk_ready_callback = chunk_ready_callback
        self._timeout_seconds = timeout_seconds
        # Per-step wall-time accumulators; each key is written by exactly one pipeline thread.
        self._timings: dict[str, float] = {}
        self._cuda_device = torch.cuda.current_device()
        self._optimizer_param_ids = {id(param) for chunk in chunks for param in chunk}
        self._chunk_param_ids = [{id(param) for param in chunk} for chunk in chunks]
        self._chunk_by_param_id = {
            param_id: chunk_idx for chunk_idx, param_ids in enumerate(self._chunk_param_ids) for param_id in param_ids
        }
        params = {id(param): param for chunk in chunks for param in chunk}
        if gradient_dtypes.keys() != params.keys() or compute_dtypes.keys() != params.keys():
            raise ValueError("Full-offload dtype policy must cover every optimizer parameter exactly")
        supported_dtypes = {torch.bfloat16, torch.float32}
        if not set(gradient_dtypes.values()) <= supported_dtypes:
            raise TypeError(f"Unsupported full-offload gradient dtypes: {set(gradient_dtypes.values())}")
        if not set(compute_dtypes.values()) <= supported_dtypes:
            raise TypeError(f"Unsupported full-offload compute dtypes: {set(compute_dtypes.values())}")
        self._gradient_dtypes = gradient_dtypes
        self._compute_dtypes = compute_dtypes
        dtensor_params = [param for param in params.values() if isinstance(param.data, DTensor)]
        if len(dtensor_params) != len(params):
            raise TypeError("Gradient offload requires FSDP2 DTensor parameters")

        alignment = 256 // torch.empty((), dtype=torch.float32).element_size()
        offsets: dict[int, int] = {}
        slab_numel = 0
        max_param_numel: dict[torch.dtype, int] = defaultdict(int)
        for param in dtensor_params:
            local = param.data.to_local()
            offsets[id(param)] = slab_numel
            slab_numel += (local.numel() + alignment - 1) // alignment * alignment
            gradient_dtype = gradient_dtypes[id(param)]
            max_param_numel[gradient_dtype] = max(max_param_numel[gradient_dtype], local.numel())
        accumulator_slab = torch.empty(slab_numel, dtype=torch.float32, device="cpu")
        self._buffers: dict[int, _CPUGradientBuffer] = {}
        self._ready_events: dict[int, list[torch.cuda.Event]] = {}
        for param in dtensor_params:
            data = param.data
            local = data.to_local()
            accumulator = accumulator_slab.narrow(0, offsets[id(param)], local.numel()).view(local.shape)
            template = copy.copy(data.detach())
            template._local_tensor = accumulator
            self._buffers[id(param)] = _CPUGradientBuffer(template, accumulator)
            self._ready_events[id(param)] = [torch.cuda.Event() for _ in range(max_inflight_backwards)]

        max_chunk_numel: dict[torch.dtype, int] = defaultdict(int)
        for chunk in chunks:
            chunk_numel: dict[torch.dtype, int] = defaultdict(int)
            for param in chunk:
                chunk_numel[compute_dtypes[id(param)]] += param.to_local().numel()
            for dtype, numel in chunk_numel.items():
                max_chunk_numel[dtype] = max(max_chunk_numel[dtype], numel)
        self._input_normal_capacity = {
            dtype: min(numel, target_chunk_numel) for dtype, numel in max_param_numel.items()
        }
        self._output_normal_capacity = {
            dtype: min(numel, target_chunk_numel) for dtype, numel in max_chunk_numel.items()
        }
        self._input_slots = {}
        self._free_input_slots = {}
        for dtype, max_numel in max_param_numel.items():
            self._input_slots[dtype], self._free_input_slots[dtype] = self._allocate_slots(
                buffer_count, self._input_normal_capacity[dtype], max_numel, dtype
            )
        self._output_slots = {}
        self._free_output_slots = {}
        for dtype, max_numel in max_chunk_numel.items():
            self._output_slots[dtype], self._free_output_slots[dtype] = self._allocate_slots(
                buffer_count, self._output_normal_capacity[dtype], max_numel, dtype
            )

        self._d2h_stream = torch.cuda.Stream()
        self._transfer_requests: queue.SimpleQueue[_GradientTransferRequest | _BackwardBoundaryTask | None] = (
            queue.SimpleQueue()
        )
        self._tasks: queue.SimpleQueue[_GradientCopyTask | _BackwardBoundaryTask | None] = queue.SimpleQueue()
        self._release_tasks: queue.SimpleQueue[_PinnedTransferSlot | None] = queue.SimpleQueue()
        self._condition = threading.Condition()
        self._gradient_scale = 1.0
        self._final_backward = False
        self._overlap_optimizer = False
        self._step_generation = 0
        self._backward_generation = 0
        self._backward_open = False
        self._backward_enqueued_param_ids: set[int] = set()
        self._pending_chunk_param_ids: list[set[int]] = []
        self._scheduled_chunks: set[int] = set()
        self._completed_chunks = 0
        self._worker_error: BaseException | None = None
        self._closed = False

        self._hook_handles = [param.register_post_accumulate_grad_hook(self._offload_hook) for param in params.values()]
        self._transfer_worker = threading.Thread(
            target=self._worker_entry,
            args=(self._transfer_loop, "gradient transfer scheduler"),
            name="grad-transfer",
            daemon=True,
        )
        self._worker = threading.Thread(
            target=self._worker_entry,
            args=(self._copy_worker_loop, "gradient copy worker"),
            name="grad-offload",
            daemon=True,
        )
        self._release_worker = threading.Thread(
            target=self._worker_entry,
            args=(self._release_loop, "weight transfer reclaimer"),
            name="weight-transfer-reclaimer",
            daemon=True,
        )
        self._transfer_worker.start()
        self._worker.start()
        self._release_worker.start()

        input_slots = [slot for slots in self._input_slots.values() for slot in slots]
        output_slots = [slot for slots in self._output_slots.values() for slot in slots]
        pinned_bytes = sum(slot.tensor.nbytes for slot in input_slots + output_slots)
        get_logger().info(
            "Native full offload uses pageable FP32 gradients and bounded mixed-dtype transfer rings "
            f"({len(input_slots)} D2H slots, {len(output_slots)} H2D slots, "
            f"{pinned_bytes / 1024**3:.2f} GiB pinned, "
            f"{accumulator_slab.nbytes / 1024**3:.2f} GiB pageable accumulator)"
        )

    @staticmethod
    def _allocate_slots(
        count: int,
        normal_numel: int,
        max_numel: int,
        dtype: torch.dtype,
    ) -> tuple[list[_PinnedTransferSlot], dict[str, queue.Queue[_PinnedTransferSlot]]]:
        slots = [
            _PinnedTransferSlot(
                index=index,
                tensor=torch.empty(normal_numel, dtype=dtype, device="cpu", pin_memory=True),
                event=torch.cuda.Event(),
            )
            for index in range(count)
        ]
        if max_numel > normal_numel:
            slots.append(
                _PinnedTransferSlot(
                    index=count,
                    tensor=torch.empty(max_numel, dtype=dtype, device="cpu", pin_memory=True),
                    event=torch.cuda.Event(),
                )
            )
        free_slots = {"normal": queue.Queue(), "oversized": queue.Queue()}
        for slot in slots:
            free_slots["normal" if slot.tensor.numel() == normal_numel else "oversized"].put(slot)
        return slots, free_slots

    @staticmethod
    def _free_slot_count(slots: dict[str, queue.Queue[_PinnedTransferSlot]]) -> int:
        return sum(slot_queue.qsize() for slot_queue in slots.values())

    @staticmethod
    def _total_free_slot_count(
        slots_by_dtype: dict[torch.dtype, dict[str, queue.Queue[_PinnedTransferSlot]]],
    ) -> int:
        return sum(GradientOffloadManager._free_slot_count(slots) for slots in slots_by_dtype.values())

    @staticmethod
    def _total_slot_count(slots_by_dtype: dict[torch.dtype, list[_PinnedTransferSlot]]) -> int:
        return sum(len(slots) for slots in slots_by_dtype.values())

    def _worker_entry(self, target: Callable[[], None], name: str) -> None:
        try:
            with torch.cuda.device(self._cuda_device):
                target()
        except BaseException as error:
            with self._condition:
                if self._worker_error is None:
                    self._worker_error = RuntimeError(f"{name} failed: {error}")
                    self._worker_error.__cause__ = error
                self._condition.notify_all()

    def _diagnostics(self) -> str:
        pending = [
            f"chunk={self._chunk_by_param_id[param_id]} step={buffer.pending_step} backward={buffer.pending_backward}"
            for param_id, buffer in self._buffers.items()
            if buffer.pending_generations
        ]
        return (
            f"step={self._step_generation}, backward={self._backward_generation}, "
            f"pending_gradients={len(pending)}, pending_sample={pending[:8]}, "
            f"completed_chunks={self._completed_chunks}/{len(self._chunks)}, "
            f"scheduled_chunks={len(self._scheduled_chunks)}, "
            f"free_d2h_slots={self._total_free_slot_count(self._free_input_slots)}/"
            f"{self._total_slot_count(self._input_slots)}, "
            f"free_h2d_slots={self._total_free_slot_count(self._free_output_slots)}/"
            f"{self._total_slot_count(self._output_slots)}"
        )

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(f"Native CPU offload pipeline failed ({self._diagnostics()})") from self._worker_error

    def _wait_for(self, predicate: Callable[[], bool], description: str) -> None:
        deadline = time.monotonic() + self._timeout_seconds
        with self._condition:
            while not predicate():
                self._raise_worker_error()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for {description} ({self._diagnostics()})")
                self._condition.wait(timeout=min(1.0, remaining))
            self._raise_worker_error()

    def _acquire_slot(
        self,
        slots: dict[str, queue.Queue[_PinnedTransferSlot]],
        direction: str,
        numel: int,
        normal_capacity: int,
    ) -> _PinnedTransferSlot:
        slot_class = "normal" if numel <= normal_capacity else "oversized"
        slot_queue = slots[slot_class]
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            self._raise_worker_error()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out acquiring a {direction} slot ({self._diagnostics()})")
            try:
                return slot_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                continue

    def _transfer_loop(self) -> None:
        while True:
            task = self._transfer_requests.get()
            if task is None:
                return
            if isinstance(task, _BackwardBoundaryTask):
                self._tasks.put(task)
                continue
            acquire_start = time.perf_counter()
            gradient_dtype = task.local_grad.dtype
            slot = self._acquire_slot(
                self._free_input_slots[gradient_dtype],
                "D2H",
                task.local_grad.numel(),
                self._input_normal_capacity[gradient_dtype],
            )
            self._timings["d2h_slot_wait"] = (
                self._timings.get("d2h_slot_wait", 0.0) + time.perf_counter() - acquire_start
            )
            if task.local_grad.numel() > slot.tensor.numel():
                raise RuntimeError(
                    f"Gradient with {task.local_grad.numel()} elements exceeds transfer slot capacity "
                    f"{slot.tensor.numel()}"
                )
            destination = slot.tensor.narrow(0, 0, task.local_grad.numel()).view(task.local_grad.shape)
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(task.ready_event)
                destination.copy_(task.local_grad, non_blocking=True)
                task.local_grad.record_stream(self._d2h_stream)
                slot.event.record(self._d2h_stream)
            self._tasks.put(_GradientCopyTask(task, slot))

    def _copy_worker_loop(self) -> None:
        deferred: _GradientCopyTask | _BackwardBoundaryTask | None = None
        stop_after_batch = False
        while True:
            task = deferred if deferred is not None else self._tasks.get()
            deferred = None
            if task is None:
                return
            if isinstance(task, _BackwardBoundaryTask):
                if task.final_backward:
                    for param_id in task.missing_param_ids:
                        self._mark_final_param_ready(param_id)
                continue
            batch = [task]
            batch_param_ids = {task.request.param_id}
            while len(batch) < self._total_slot_count(self._input_slots):
                try:
                    next_task = self._tasks.get_nowait()
                except queue.Empty:
                    break
                if next_task is None:
                    stop_after_batch = True
                    break
                if isinstance(next_task, _BackwardBoundaryTask):
                    deferred = next_task
                    break
                if next_task.request.param_id in batch_param_ids:
                    deferred = next_task
                    break
                batch.append(next_task)
                batch_param_ids.add(next_task.request.param_id)

            sync_start = time.perf_counter()
            batch[-1].slot.event.synchronize()
            materialize_start = time.perf_counter()
            self._timings["d2h_wait"] = self._timings.get("d2h_wait", 0.0) + materialize_start - sync_start
            buffers = [self._buffers[item.request.param_id] for item in batch]
            sources = [
                item.slot.tensor.narrow(0, 0, item.request.local_grad.numel()).view(item.request.local_grad.shape)
                for item in batch
            ]
            bfloat16_indices = [index for index, source in enumerate(sources) if source.dtype == torch.bfloat16]
            if bfloat16_indices:
                native_copy_or_add_bfloat16_multi_(
                    [buffers[index].accumulator for index in bfloat16_indices],
                    [sources[index] for index in bfloat16_indices],
                    [buffers[index].initialized for index in bfloat16_indices],
                )
            for buffer, source in zip(buffers, sources):
                if source.dtype == torch.float32:
                    if buffer.initialized:
                        buffer.accumulator.add_(source)
                    else:
                        buffer.accumulator.copy_(source)
            self._timings["materialize"] = (
                self._timings.get("materialize", 0.0) + time.perf_counter() - materialize_start
            )
            for item in batch:
                dtype = item.slot.tensor.dtype
                slot_class = "normal" if item.slot.tensor.numel() == self._input_normal_capacity[dtype] else "oversized"
                self._free_input_slots[dtype][slot_class].put(item.slot)
            with self._condition:
                for item, buffer in zip(batch, buffers):
                    request = item.request
                    generation = (request.step, request.backward)
                    if generation not in buffer.pending_generations:
                        raise RuntimeError(
                            f"Gradient generation {generation} disappeared before its transfer completed"
                        )
                    buffer.pending_generations.remove(generation)
                    buffer.initialized = True
                    buffer.pending = bool(buffer.pending_generations)
                    if buffer.pending_generations:
                        buffer.pending_step, buffer.pending_backward = min(buffer.pending_generations)
                    else:
                        buffer.pending_step = -1
                        buffer.pending_backward = -1
                self._condition.notify_all()
            for item in batch:
                if item.request.final_backward:
                    self._mark_final_param_ready(item.request.param_id)
            if stop_after_batch:
                return

    def _release_loop(self) -> None:
        while True:
            slot = self._release_tasks.get()
            if slot is None:
                return
            slot.event.synchronize()
            dtype = slot.tensor.dtype
            slot_class = "normal" if slot.tensor.numel() == self._output_normal_capacity[dtype] else "oversized"
            self._free_output_slots[dtype][slot_class].put(slot)
            with self._condition:
                self._condition.notify_all()

    def _mark_final_param_ready(self, param_id: int) -> None:
        chunk_idx = self._chunk_by_param_id[param_id]
        should_run = False
        with self._condition:
            pending = self._pending_chunk_param_ids[chunk_idx]
            if param_id not in pending:
                raise RuntimeError(
                    f"Parameter in chunk {chunk_idx} became ready more than once "
                    f"at step {self._step_generation}, backward {self._backward_generation}"
                )
            pending.remove(param_id)
            if not pending and chunk_idx not in self._scheduled_chunks:
                self._scheduled_chunks.add(chunk_idx)
                should_run = True
        if should_run:
            self._run_optimizer_chunk(chunk_idx)

    def _run_optimizer_chunk(self, chunk_idx: int) -> None:
        chunk_start = time.perf_counter()
        self._chunk_ready_callback(chunk_idx)
        self._timings["optimizer_chunk"] = self._timings.get("optimizer_chunk", 0.0) + time.perf_counter() - chunk_start
        with self._condition:
            self._completed_chunks += 1
            self._condition.notify_all()

    def consume_timings(self) -> dict[str, float]:
        timings, self._timings = self._timings, {}
        return timings

    def begin_step(self, gradient_scale: float, *, overlap_optimizer: bool) -> None:
        self.wait()
        with self._condition:
            self._raise_worker_error()
            if self._backward_open:
                raise RuntimeError("Cannot begin an optimizer step while a backward is open")
            self._step_generation += 1
            self._gradient_scale = gradient_scale
            self._overlap_optimizer = overlap_optimizer
            self._final_backward = False
            self._pending_chunk_param_ids = []
            self._scheduled_chunks.clear()
            self._completed_chunks = 0

    def begin_backward(self, *, final_backward: bool) -> None:
        with self._condition:
            self._raise_worker_error()
            if self._backward_open:
                raise RuntimeError("begin_backward called before the previous backward finished")
            self._backward_generation += 1
            self._backward_open = True
            self._final_backward = final_backward
            self._backward_enqueued_param_ids.clear()
            if final_backward and self._overlap_optimizer:
                self._pending_chunk_param_ids = [set(param_ids) for param_ids in self._chunk_param_ids]
                self._scheduled_chunks.clear()
                self._completed_chunks = 0

    @torch.no_grad()
    def _offload_params(self, params: list[nn.Parameter]) -> None:
        for param in params:
            param_id = id(param)
            if param_id not in self._optimizer_param_ids or param.grad is None:
                continue
            if not isinstance(param.grad, DTensor):
                raise TypeError(f"Expected FSDP2 DTensor gradient, got {type(param.grad)}")
            local_grad = param.grad.to_local()
            expected_dtype = self._gradient_dtypes[param_id]
            if local_grad.dtype != expected_dtype:
                raise TypeError(
                    f"Full-offload gradient dtype mismatch in chunk {self._chunk_by_param_id[param_id]}: "
                    f"expected {expected_dtype}, got {local_grad.dtype}"
                )
            buffer = self._buffers[param_id]
            with self._condition:
                self._raise_worker_error()
                if not self._backward_open:
                    raise RuntimeError("Gradient hook ran outside begin_backward/finish_backward")
                if buffer.last_enqueued_backward == self._backward_generation:
                    raise RuntimeError(
                        f"Parameter in chunk {self._chunk_by_param_id[param_id]} produced more than one gradient "
                        f"in backward {self._backward_generation}"
                    )
                generation = (self._step_generation, self._backward_generation)
                event_index = (self._backward_generation - 1) % len(self._ready_events[param_id])
                if any(
                    (pending_backward - 1) % len(self._ready_events[param_id]) == event_index
                    for _, pending_backward in buffer.pending_generations
                ):
                    raise RuntimeError(
                        f"Gradient event window exhausted before an older contribution drained ({self._diagnostics()})"
                    )
                buffer.pending = True
                buffer.pending_generations.add(generation)
                buffer.pending_step, buffer.pending_backward = min(buffer.pending_generations)
                buffer.last_enqueued_backward = self._backward_generation
                self._backward_enqueued_param_ids.add(param_id)
            ready_event = self._ready_events[param_id][event_index]
            ready_event.record(torch.cuda.current_stream())
            self._transfer_requests.put(
                _GradientTransferRequest(
                    param_id=param_id,
                    local_grad=local_grad,
                    ready_event=ready_event,
                    step=self._step_generation,
                    backward=self._backward_generation,
                    final_backward=self._final_backward and self._overlap_optimizer,
                )
            )
            param.grad = None

    def _offload_hook(self, param: torch.Tensor) -> None:
        if not isinstance(param, nn.Parameter):
            raise TypeError(f"Expected parameter in post-accumulate hook, got {type(param)}")
        self._offload_params([param])

    def finish_backward(self, *, wait_for_copies: bool = True) -> None:
        remaining = [param for chunk in self._chunks for param in chunk if param.grad is not None]
        self._offload_params(remaining)
        with self._condition:
            if not self._backward_open:
                raise RuntimeError("finish_backward called without begin_backward")
            missing = (
                self._optimizer_param_ids - self._backward_enqueued_param_ids
                if self._final_backward and self._overlap_optimizer
                else set()
            )
            boundary = _BackwardBoundaryTask(
                step=self._step_generation,
                backward=self._backward_generation,
                final_backward=self._final_backward and self._overlap_optimizer,
                missing_param_ids=missing,
            )
            self._backward_open = False
        self._transfer_requests.put(boundary)
        if wait_for_copies:
            self.wait()

    def wait(self) -> None:
        self._wait_for(
            lambda: not any(buffer.pending_generations for buffer in self._buffers.values()),
            "gradient transfers",
        )

    def wait_for_optimizer(self) -> None:
        if self._overlap_optimizer:
            self._wait_for(lambda: self._completed_chunks == len(self._chunks), "CPU optimizer chunks")

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

    def load_cpu_chunk(self, chunk_idx: int, *, wait: bool = True) -> list[torch.Tensor | None]:
        if wait:
            self.wait()
        gradients: list[torch.Tensor | None] = []
        for param in self._chunks[chunk_idx]:
            buffer = self._buffers[id(param)]
            gradients.append(buffer.accumulator if buffer.initialized else None)
        return gradients

    def acquire_output_chunk(self, chunk_idx: int) -> tuple[list[_PinnedTransferSlot], list[torch.Tensor]]:
        chunk_numel: dict[torch.dtype, int] = defaultdict(int)
        for param in self._chunks[chunk_idx]:
            chunk_numel[self._compute_dtypes[id(param)]] += param.to_local().numel()
        slots = {
            dtype: self._acquire_slot(
                self._free_output_slots[dtype],
                "H2D",
                numel,
                self._output_normal_capacity[dtype],
            )
            for dtype, numel in chunk_numel.items()
        }
        offsets: dict[torch.dtype, int] = defaultdict(int)
        views = []
        for param in self._chunks[chunk_idx]:
            local = param.to_local()
            dtype = self._compute_dtypes[id(param)]
            views.append(slots[dtype].tensor.narrow(0, offsets[dtype], local.numel()).view(local.shape))
            offsets[dtype] += local.numel()
        return list(slots.values()), views

    def release_output_chunk(self, slots: list[_PinnedTransferSlot], stream: torch.cuda.Stream) -> None:
        for slot in slots:
            slot.event.record(stream)
            self._release_tasks.put(slot)

    def wait_for_output_slots(self) -> None:
        self._wait_for(
            lambda: self._total_free_slot_count(self._free_output_slots) == self._total_slot_count(self._output_slots),
            "H2D transfer slots",
        )

    def zero_grad(self) -> None:
        self.wait()
        for buffer in self._buffers.values():
            buffer.initialized = False
            buffer.pending_generations.clear()
        self._gradient_scale = 1.0

    def close(self) -> None:
        if self._closed:
            return
        self.wait_for_optimizer()
        self.wait()
        self.wait_for_output_slots()
        for handle in self._hook_handles:
            handle.remove()
        self._transfer_requests.put(None)
        self._transfer_worker.join(timeout=self._timeout_seconds)
        self._tasks.put(None)
        self._worker.join(timeout=self._timeout_seconds)
        self._release_tasks.put(None)
        self._release_worker.join(timeout=self._timeout_seconds)
        if self._transfer_worker.is_alive() or self._worker.is_alive() or self._release_worker.is_alive():
            raise TimeoutError(f"Timed out closing native CPU offload workers ({self._diagnostics()})")
        self._closed = True


class FullCPUOffloadOptimizer(OffloadOptimizer):
    """Runs the optimizer on CPU-resident FP32 masters, overlapped with backward.

    The GPU keeps persistent mixed-precision compute parameters; FP32 masters,
    moments, and accumulated gradients live in CPU RAM, each optimizer chunk
    runs as soon as its last gradient arrives, and refreshed weights stream back
    while backward is still executing.
    """

    _TARGET_CPU_CHUNK_NUMEL = 16 * 1024**2
    _TRANSFER_BUFFER_COUNT = 4
    _MAX_INFLIGHT_BACKWARDS = 16
    _TIMEOUT_SECONDS = 120.0
    _MASTER_WEIGHT_STATE = "prime_rl_master_weight"

    def __init__(
        self,
        optimizer: Optimizer,
        master_weights: dict[int, _MasterWeight],
        dp_replicate: int = 1,
        muon_matrix_partitions: dict[nn.Parameter, tuple[int, ...]] | None = None,
    ):
        self.optimizer = optimizer
        self._muon_matrix_partitions = muon_matrix_partitions or {}
        self._initialized = False
        self._master_weights = master_weights
        self._chunks = self._build_chunks()
        # Reuse the transfer streams across steps: fresh streams each step land
        # their H2D/D2H staging in new per-stream allocator pools, growing
        # reserved memory every step and starving the default stream.
        self._h2d_stream = torch.cuda.Stream()
        self._d2h_stream = torch.cuda.Stream()
        self._cuda_device = torch.cuda.current_device()
        self._chunk_groups = self._build_chunk_groups()
        self._adamw = isinstance(optimizer, AdamW)
        self._muon = isinstance(optimizer, Muon)
        self._muon_chunks = {
            chunk_idx
            for chunk_idx, groups in enumerate(self._chunk_groups)
            if self._muon and groups[0][0]["algorithm"] == "muon"
        }
        self._muon_params_with_grad: set[int] = set()
        self._muon_slots: list[tuple[torch.Tensor, torch.Tensor, torch.cuda.Event, torch.cuda.Event]] = []
        get_logger().info(f"Loading native read-only-gradient multi-tensor CPU {type(optimizer).__name__} kernel")
        load_cpu_adamw_kernel()
        self._gradient_manager = GradientOffloadManager(
            [[master_weights[id(param)].model_param for param in chunk] for chunk in self._chunks],
            dp_replicate,
            buffer_count=self._TRANSFER_BUFFER_COUNT,
            max_inflight_backwards=self._MAX_INFLIGHT_BACKWARDS,
            timeout_seconds=self._TIMEOUT_SECONDS,
            target_chunk_numel=self._TARGET_CPU_CHUNK_NUMEL,
            chunk_ready_callback=self._step_cpu_chunk,
            gradient_dtypes={id(master.model_param): master.gradient_dtype for master in master_weights.values()},
            compute_dtypes={id(master.model_param): master.compute_dtype for master in master_weights.values()},
        )
        self._initialize_cpu_optimizer_state()

    def _build_chunks(self) -> list[list[nn.Parameter]]:
        """Group optimizer parameters into bounded, model-independent chunks."""
        target_numel = self._TARGET_CPU_CHUNK_NUMEL
        chunks: list[list[nn.Parameter]] = []
        chunk: list[nn.Parameter] = []
        chunk_numel = 0
        for group in self.optimizer.param_groups:
            # Chunks never span param groups, so a Muon chunk holds only Muon matrices
            if chunk:
                chunks.append(chunk)
                chunk = []
                chunk_numel = 0
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

    @torch.no_grad()
    def _update_compute_weights(
        self,
        chunk_idx: int,
        stream: torch.cuda.Stream,
        sources: list[torch.Tensor],
    ) -> None:
        assert self._master_weights is not None
        with torch.cuda.stream(stream):
            for param_idx, param in enumerate(self._chunks[chunk_idx]):
                master = self._master_weights[id(param)]
                if not isinstance(master.model_param, DTensor):
                    raise TypeError(f"Expected FSDP2 DTensor parameter, got {type(master.model_param)}")
                source = sources[param_idx]
                local_param = master.model_param.to_local()
                if local_param.dtype != master.compute_dtype:
                    raise TypeError(
                        f"Full-offload compute dtype changed: expected {master.compute_dtype}, got {local_param.dtype}"
                    )
                if source.dtype != master.compute_dtype:
                    raise TypeError(
                        f"Full-offload H2D source dtype mismatch: expected {master.compute_dtype}, got {source.dtype}"
                    )
                local_param.copy_(source, non_blocking=True)

    def _step_native_cpu_adamw_chunk(
        self,
        chunk_idx: int,
        gradients: list[torch.Tensor | None],
        compute_params: list[torch.Tensor],
    ) -> None:
        assert self._gradient_manager is not None
        assert self._master_weights is not None
        gradient_by_param_id = {
            id(param): gradient for param, gradient in zip(self._chunks[chunk_idx], gradients) if gradient is not None
        }
        compute_by_param_id = {
            id(param): compute_param for param, compute_param in zip(self._chunks[chunk_idx], compute_params)
        }
        for param, gradient, compute_param in zip(self._chunks[chunk_idx], gradients, compute_params):
            if gradient is None:
                compute_param.copy_(self._master_weights[id(param)].cpu_tensor)
        for group, params in self._chunk_groups[chunk_idx]:
            params_with_grad = [param for param in params if id(param) in gradient_by_param_id]
            if not params_with_grad:
                continue
            states = [self.optimizer.state[param] for param in params_with_grad]
            group_compute_params = [compute_by_param_id[id(param)] for param in params_with_grad]
            if self._muon:
                # Dion's AdamW keeps its step count on the group and bumps it in step(), after the chunks ran
                exp_avgs = [state["momentum"] for state in states]
                exp_avg_sqs = [state["variance"] for state in states]
                steps = [torch.tensor(float(group["step"])) for _ in states]
                beta1, beta2, eps = group["beta1"], group["beta2"], group["epsilon"]
            else:
                exp_avgs = [state["exp_avg"] for state in states]
                exp_avg_sqs = [state["exp_avg_sq"] for state in states]
                steps = [state["step"] for state in states]
                (beta1, beta2), eps = group["betas"], group["eps"]
            native_cpu_adamw_step(
                params_with_grad,
                [gradient_by_param_id[id(param)] for param in params_with_grad],
                exp_avgs,
                exp_avg_sqs,
                steps,
                group_compute_params,
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                weight_decay=group["weight_decay"],
                eps=eps,
                gradient_scale=self._gradient_manager.gradient_scale,
            )

    def _step_native_cpu_sign_sgd_chunk(
        self,
        chunk_idx: int,
        gradients: list[torch.Tensor | None],
        compute_params: list[torch.Tensor],
    ) -> None:
        assert self._gradient_manager is not None
        assert self._master_weights is not None
        # The kernel consumes unscaled gradients: sign(scale * g) == sign(g) for scale > 0,
        # so the positive gradient scale can be dropped entirely.
        assert self._gradient_manager.gradient_scale > 0
        gradient_by_param_id = {
            id(param): gradient for param, gradient in zip(self._chunks[chunk_idx], gradients) if gradient is not None
        }
        compute_by_param_id = {
            id(param): compute_param for param, compute_param in zip(self._chunks[chunk_idx], compute_params)
        }
        for param, gradient, compute_param in zip(self._chunks[chunk_idx], gradients, compute_params):
            if gradient is None:
                compute_param.copy_(self._master_weights[id(param)].cpu_tensor)
        for group, params in self._chunk_groups[chunk_idx]:
            params_with_grad = [param for param in params if id(param) in gradient_by_param_id]
            if not params_with_grad:
                continue
            native_cpu_sign_sgd_step(
                params_with_grad,
                [gradient_by_param_id[id(param)] for param in params_with_grad],
                [compute_by_param_id[id(param)] for param in params_with_grad],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
            )

    def _step_native_cpu_chunk(
        self,
        chunk_idx: int,
        gradients: list[torch.Tensor | None],
        compute_params: list[torch.Tensor],
    ) -> None:
        if self._adamw or self._muon:
            self._step_native_cpu_adamw_chunk(chunk_idx, gradients, compute_params)
        else:
            self._step_native_cpu_sign_sgd_chunk(chunk_idx, gradients, compute_params)

    @torch.no_grad()
    def _update_muon_momentum(self, chunk_idx: int, gradients: list[torch.Tensor | None]) -> None:
        """Fold this step's gradients into the FP32 CPU momentum; Newton-Schulz runs in step()."""
        gradient_by_param_id = {
            id(param): gradient for param, gradient in zip(self._chunks[chunk_idx], gradients) if gradient is not None
        }
        for group, params in self._chunk_groups[chunk_idx]:
            if group["nesterov"]:
                raise NotImplementedError("Full offload does not support Nesterov Muon")
            for param in params:
                if id(param) not in gradient_by_param_id:
                    continue
                momentum = self.optimizer.state[param]["momentum"]
                momentum.mul_(group["mu"]).add_(
                    gradient_by_param_id[id(param)], alpha=self._gradient_manager.gradient_scale
                )
                self._muon_params_with_grad.add(id(param))

    def _step_cpu_chunk(self, chunk_idx: int) -> None:
        gradients = self._gradient_manager.load_cpu_chunk(chunk_idx, wait=False)
        timings = self._gradient_manager._timings
        if chunk_idx in self._muon_chunks:
            kernel_start = time.perf_counter()
            self._update_muon_momentum(chunk_idx, gradients)
            timings["muon_momentum"] = timings.get("muon_momentum", 0.0) + time.perf_counter() - kernel_start
            return
        acquire_start = time.perf_counter()
        output_slots, compute_params = self._gradient_manager.acquire_output_chunk(chunk_idx)
        kernel_start = time.perf_counter()
        timings["output_slot_wait"] = timings.get("output_slot_wait", 0.0) + kernel_start - acquire_start
        self._step_native_cpu_chunk(chunk_idx, gradients, compute_params)
        timings["optimizer_kernel"] = timings.get("optimizer_kernel", 0.0) + time.perf_counter() - kernel_start
        with torch.cuda.device(self._cuda_device):
            self._update_compute_weights(chunk_idx, self._h2d_stream, compute_params)
            self._gradient_manager.release_output_chunk(output_slots, self._h2d_stream)

    def _muon_batches(self, group: dict, cpu_params: list[nn.Parameter]):
        """Yield Dion's batches of same-shaped matrices as (model params, CPU masters, mesh info)."""
        mesh_info = self.optimizer._get_group_mesh_info(group)
        cpu_by_model_param = {id(self._master_weights[id(param)].model_param): param for param in cpu_params}
        partitions = self._muon_matrix_partitions
        model_params = [self._master_weights[id(param)].model_param for param in cpu_params]
        for batch in create_param_batches(model_params, batch_size=mesh_info[1], matrix_partitions=partitions):
            yield batch, [cpu_by_model_param[id(param)] for param in batch], partitions.get(batch[0]), mesh_info

    def _muon_transfer_slots(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.cuda.Event, torch.cuda.Event]]:
        if not self._muon_slots:
            max_numel = max_bytes = 0
            for group in self.optimizer.param_groups:
                if group["algorithm"] != "muon":
                    continue
                for batch, _, _, _ in self._muon_batches(group, group["params"]):
                    locals_ = [param.to_local() for param in batch]
                    max_numel = max(max_numel, sum(local.numel() for local in locals_))
                    max_bytes = max(max_bytes, sum(local.nbytes for local in locals_))
            self._muon_slots = [
                (
                    torch.empty(max_numel, dtype=torch.bfloat16, pin_memory=True),
                    torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True),
                    torch.cuda.Event(),
                    torch.cuda.Event(),
                )
                for _ in range(2)
            ]
        return self._muon_slots

    @torch.no_grad()
    def _step_muon_matrices(self) -> None:
        """Orthogonalize the CPU momentum on GPU and apply the update to the CPU masters.

        Per batch of same-shaped matrices, BF16 momentum goes H2D and Dion's batched Newton-Schulz
        runs on it, including its all-to-all for FSDP-sharded matrices. Dion gets a zero gradient,
        unit momentum, and a zero parameter, so its in-place update leaves exactly
        -adjusted_lr * orthogonalized momentum, which goes D2H and is added to the FP32 master.
        Two batches are in flight, so transfers and CPU work overlap Newton-Schulz.
        """
        slots = self._muon_transfer_slots()
        pending = []
        index = 0
        for group in self.optimizer.param_groups:
            group["step"] += 1
            if group["algorithm"] != "muon":
                continue
            cpu_params = [param for param in group["params"] if id(param) in self._muon_params_with_grad]
            for batch, batch_cpu_params, partitions, mesh_info in self._muon_batches(group, cpu_params):
                device_rank, world_size, process_group, _ = mesh_info
                momentum_buffer, update_buffer, h2d_done, d2h_done = slots[index % len(slots)]
                index += 1
                h2d_done.synchronize()
                shard_dim, is_local = self.optimizer._get_matrix_shard_dim(batch[0], group["flatten"], process_group)

                momentums, updates, update_views = [], [], []
                numel_offset = byte_offset = 0
                self._h2d_stream.wait_stream(torch.cuda.current_stream())
                for model_param, cpu_param in zip(batch, batch_cpu_params):
                    local = model_param.to_local()
                    staged = momentum_buffer.narrow(0, numel_offset, local.numel()).view(local.shape)
                    staged.copy_(self.optimizer.state[cpu_param]["momentum"])
                    numel_offset += local.numel()
                    update_views.append(
                        update_buffer.narrow(0, byte_offset, local.nbytes).view(local.dtype).view(local.shape)
                    )
                    byte_offset += local.nbytes
                    momentum = torch.empty_like(local, dtype=torch.bfloat16)
                    with torch.cuda.stream(self._h2d_stream):
                        momentum.copy_(staged, non_blocking=True)
                    momentums.append(momentum)
                    updates.append(DTensor(torch.zeros_like(local), model_param._spec, requires_grad=False))
                torch.cuda.current_stream().wait_stream(self._h2d_stream)
                gradients = [update.to_local() for update in updates]
                if not is_local:
                    updates = pad_batch(updates, world_size)
                    gradients = pad_batch(gradients, world_size)
                    momentums = pad_batch(momentums, world_size)
                for _ in muon_update_batch_async(
                    X=updates,
                    G=gradients,
                    M=momentums,
                    lr=torch.tensor(group["lr"]),
                    momentum=torch.tensor(1.0),
                    weight_decay=torch.tensor(0.0),
                    epsilon=torch.tensor(group["epsilon"]),
                    nesterov=False,
                    flatten=group["flatten"],
                    adjust_lr=group["adjust_lr"],
                    device_rank=device_rank,
                    world_size=world_size,
                    shard_dim=shard_dim,
                    is_local=is_local,
                    process_group=process_group,
                    newton_schulz_func=self.optimizer._newton_schulz_func,
                    partitions=partitions,
                ):
                    pass
                self._d2h_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._d2h_stream):
                    for update, view in zip(updates, update_views):
                        view.copy_(update.to_local(), non_blocking=True)
                        update.to_local().record_stream(self._d2h_stream)
                d2h_done.record(self._d2h_stream)
                pending.append((group, batch, batch_cpu_params, update_views, h2d_done, d2h_done))
                if len(pending) == len(slots):
                    self._apply_muon_update(*pending.pop(0))
        for args in pending:
            self._apply_muon_update(*args)
        self._muon_params_with_grad.clear()

    def _apply_muon_update(self, group, batch, cpu_params, update_views, h2d_done, d2h_done) -> None:
        d2h_done.synchronize()
        decay = 1 - group["lr"] * group["weight_decay"]
        for model_param, cpu_param, view in zip(batch, cpu_params, update_views):
            if decay != 1:
                cpu_param.mul_(decay)
            if view.dtype == torch.bfloat16:
                native_add_bfloat16_(cpu_param, view)
            else:
                cpu_param.add_(view)
            view.copy_(cpu_param)
            with torch.cuda.stream(self._h2d_stream):
                model_param.to_local().copy_(view, non_blocking=True)
        h2d_done.record(self._h2d_stream)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Optimizer closures are not supported with CPU optimizer offload")
        if self._gradient_manager.optimizer_overlapped:
            drain_start = time.perf_counter()
            self._gradient_manager.wait_for_optimizer()
            timings = self._gradient_manager.consume_timings()
            timings["drain"] = time.perf_counter() - drain_start
            if self._muon:
                muon_start = time.perf_counter()
                self._step_muon_matrices()
                timings["muon_matrices"] = time.perf_counter() - muon_start
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
            get_logger().debug(
                "Offload pipeline: " + " ".join(f"{key}={value:.3f}s" for key, value in sorted(timings.items()))
            )
            # LRScheduler normally sets this marker through its optimizer.step wrapper.
            self.optimizer._opt_called = True
            self._initialized = True
            return
        # Synchronous path (validation steps and checkpoint boundaries).
        gradients_by_chunk = [self._gradient_manager.load_cpu_chunk(i) for i in range(len(self._chunks))]
        for i in range(len(self._chunks)):
            if i in self._muon_chunks:
                self._update_muon_momentum(i, gradients_by_chunk[i])
                continue
            output_slots, compute_params = self._gradient_manager.acquire_output_chunk(i)
            self._step_native_cpu_chunk(i, gradients_by_chunk[i], compute_params)
            self._update_compute_weights(i, self._h2d_stream, compute_params)
            self._gradient_manager.release_output_chunk(output_slots, self._h2d_stream)
        if self._muon:
            self._step_muon_matrices()
        torch.cuda.current_stream().wait_stream(self._h2d_stream)
        torch.cuda.synchronize()
        self.optimizer._opt_called = True
        self._initialized = True

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        self._gradient_manager.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
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

    @torch.no_grad()
    def _initialize_cpu_optimizer_state(self) -> None:
        if isinstance(self.optimizer, SignSGD):
            # SignSGD is stateless: no moments or step counters to materialize.
            return
        if self.optimizer.state:
            return
        if self._muon:
            # Dion's state layout: Muon matrices keep momentum, its AdamW parameters add variance
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    state = {"momentum": torch.zeros_like(param)}
                    if group["algorithm"] == "adamw":
                        state["variance"] = torch.zeros_like(param)
                    self.optimizer.state[param] = state
            return
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
            for i in range(len(self._chunks)):
                output_slots, compute_params = self._gradient_manager.acquire_output_chunk(i)
                for param, compute_param in zip(self._chunks[i], compute_params):
                    compute_param.copy_(self._master_weights[id(param)].cpu_tensor)
                self._update_compute_weights(i, self._h2d_stream, compute_params)
                self._gradient_manager.release_output_chunk(output_slots, self._h2d_stream)
            torch.cuda.current_stream().wait_stream(self._h2d_stream)
            torch.cuda.synchronize()
            self._gradient_manager.wait_for_output_slots()
        self._initialized = True

    @torch.no_grad()
    def finish_model_only_checkpoint_load(self) -> None:
        if self._master_weights is None:
            raise RuntimeError("Full optimizer offload has no master weights")
        for master in self._master_weights.values():
            if not isinstance(master.model_param, DTensor):
                raise TypeError(f"Expected FSDP2 DTensor parameter, got {type(master.model_param)}")
            master.cpu_tensor.copy_(master.model_param.to_local(), non_blocking=True)
        torch.cuda.synchronize()
        self._initialized = True


@torch.no_grad()
def _cast_full_offload_compute_parameters(
    model: nn.Module,
    dtype_policy: dict[int, tuple[torch.dtype, torch.dtype]] | None,
) -> None:
    floating_params = {
        id(param): (name, param) for name, param in model.named_parameters() if param.is_floating_point()
    }
    if dtype_policy is not None and dtype_policy.keys() != floating_params.keys():
        raise ValueError("Full-offload dtype policy must cover every floating-point model parameter exactly")
    fp32_param_ids = {
        param_id
        for param_id in floating_params
        if dtype_policy is not None and dtype_policy[param_id][0] == torch.float32
    }
    fp32_snapshots = {param_id: floating_params[param_id][1].detach().clone() for param_id in fp32_param_ids}
    original_buffers = {name: buffer.detach() for name, buffer in model.named_buffers() if buffer.is_floating_point()}
    fp32_modules = []
    covered_fp32_param_ids = set()
    for module in model.modules():
        direct_param_ids = {id(param) for param in module.parameters(recurse=False) if param.is_floating_point()}
        module_fp32_param_ids = direct_param_ids & fp32_param_ids
        if not module_fp32_param_ids:
            continue
        if direct_param_ids != module_fp32_param_ids:
            raise ValueError("Full offload cannot preserve mixed compute dtypes within one parameter-owning module")
        fp32_modules.append(module)
        covered_fp32_param_ids.update(module_fp32_param_ids)
    if covered_fp32_param_ids != fp32_param_ids:
        raise ValueError("Full offload could not find an owning module for every FP32 parameter")

    model.to(dtype=torch.bfloat16)
    for module in fp32_modules:
        module.to(dtype=torch.float32)
    for param_id, snapshot in fp32_snapshots.items():
        floating_params[param_id][1].copy_(snapshot)
    for name, buffer in model.named_buffers():
        if name in original_buffers:
            buffer.data = original_buffers[name]
    for param_id, (name, model_param) in floating_params.items():
        compute_dtype = dtype_policy[param_id][0] if dtype_policy is not None else torch.bfloat16
        if model_param.dtype != compute_dtype:
            raise TypeError(f"Failed to cast {name} to its full-offload compute dtype {compute_dtype}")


@torch.no_grad()
def _create_cpu_master_weights(
    model: nn.Module,
    named_params: list[tuple[str, nn.Parameter]],
    *,
    dtype_policy: dict[int, tuple[torch.dtype, torch.dtype]] | None = None,
) -> tuple[list[tuple[str, nn.Parameter]], dict[int, _MasterWeight]]:
    trainable_params = [(name, model_param) for name, model_param in named_params if model_param.requires_grad]
    if not trainable_params:
        raise ValueError("Gradient CPU offload requires trainable parameters")
    alignment = 256 // torch.empty((), dtype=torch.float32).element_size()
    offsets = {}
    slab_numel = 0
    for name, model_param in trainable_params:
        if not isinstance(model_param, DTensor):
            raise TypeError(f"Expected FSDP2 DTensor parameter, got {type(model_param)}")
        offsets[name] = slab_numel
        local_numel = model_param.to_local().numel()
        slab_numel += (local_numel + alignment - 1) // alignment * alignment
    master_slab = torch.empty(slab_numel, dtype=torch.float32, device="cpu")
    master_named_params = []
    master_weights = {}
    for name, model_param in trainable_params:
        local_param = model_param.to_local()
        cpu_tensor = master_slab.narrow(0, offsets[name], local_param.numel()).view(local_param.shape)
        cpu_tensor.copy_(local_param, non_blocking=True)
        master_param = nn.Parameter(cpu_tensor, requires_grad=True)
        master_named_params.append((name, master_param))
        compute_dtype, gradient_dtype = (
            dtype_policy[id(model_param)] if dtype_policy is not None else (torch.bfloat16, torch.bfloat16)
        )
        if compute_dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"Full offload does not support {compute_dtype} compute parameters ({name})")
        if gradient_dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"Full offload does not support {gradient_dtype} gradients ({name})")
        master_weights[id(master_param)] = _MasterWeight(model_param, cpu_tensor, compute_dtype, gradient_dtype)
    torch.cuda.synchronize()

    _cast_full_offload_compute_parameters(model, dtype_policy)
    del local_param
    torch.cuda.empty_cache()

    master_allocation = sum(master.cpu_tensor.nbytes for master in master_weights.values())
    get_logger().info(
        f"CPU optimizer step allocated {master_allocation / 1024**3:.2f} GiB of "
        "pageable FP32 masters; persistent GPU parameters use configured compute dtypes"
    )
    return master_named_params, master_weights
