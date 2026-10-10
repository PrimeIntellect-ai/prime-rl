"""Pipeline parallelism: the decoder layers are split into stages of consecutive layers, and each
pipeline rank holds one or more stages (one model part per stage).

A model opts in with `prune_to_pipeline_stage(units, first=..., last=...)`, which drops what the
stage does not hold (`units` are half layers, see `stage_layer_units`), and `pipeline_stage_forward(*tensors)`, which runs the stage on one
micro-batch: the first stage takes `(input_ids, position_ids, labels)` and the last one returns
the summed loss. Stages are built before FSDP and expert parallelism, which then apply to each
stage over its own ranks.
"""

import copy
import ctypes
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleZBVZeroBubble,
)

from prime_rl.configs.trainer import PipelineActivationOffloadConfig
from prime_rl.trainer.models.layers.expert_compute import defer_weight_grads
from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.pipeline_offload import PipelineActivationOffloader, offload_plan

SINGLE_STAGE_SCHEDULES: dict[str, type[PipelineScheduleSingle]] = {"1F1B": Schedule1F1B, "GPipe": ScheduleGPipe}
MULTI_STAGE_SCHEDULES: dict[str, type[PipelineScheduleMulti]] = {
    "Interleaved1F1B": ScheduleInterleaved1F1B,
    "InterleavedZeroBubble": ScheduleInterleavedZeroBubble,
    "ZBVZeroBubble": ScheduleZBVZeroBubble,
}
# Schedules whose stages zig-zag over the ranks: rank r holds stages r and 2 * pp - 1 - r.
V_SCHEDULES = {"ZBVZeroBubble", "DualPipeV"}


class StageLayers(nn.ModuleList):
    """A stage's decoder layers, registered under their indices in the full model so parameter
    names (and checkpoint keys) match the unsplit model. Integer indexing is positional."""

    def __init__(self, layers: dict[int, nn.Module]):
        super().__init__()
        for idx, layer in layers.items():
            self.add_module(str(idx), layer)

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        self._modules[list(self._modules)[idx]] = module


class PooledRecvPipelineStage(PipelineStage):
    """A `PipelineStage` whose per-micro-batch receive buffers cycle through `pool` sets for the
    activations and `grad_pool` sets (default `pool`) for their gradients instead of one set per
    micro-batch, so memory does not grow with the number of micro-batches. Only valid for schedules
    that keep fewer than `pool` micro-batches of a stage between their activation receive and their
    backward (1F1B keeps at most the number of stages)."""

    def __init__(self, *args, pool: int, grad_pool: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool, self.grad_pool = pool, grad_pool or pool

    def _setup_forward_recv_info(self, num_microbatches: int, has_backward: bool) -> None:
        super()._setup_forward_recv_info(min(self.pool, num_microbatches), has_backward)
        self._share_buffers(self.args_recv_info, self.pool, num_microbatches)

    def _setup_backward_recv_info(self, num_microbatches: int) -> None:
        super()._setup_backward_recv_info(min(self.grad_pool, num_microbatches))
        self.chunks = num_microbatches
        self._share_buffers(self.grad_recv_info, self.grad_pool, num_microbatches)

    @staticmethod
    def _share_buffers(recv_infos: dict, pool: int, num_microbatches: int) -> None:
        for chunk in range(pool, num_microbatches):
            recv_infos[chunk] = tuple(copy.copy(info) for info in recv_infos[chunk % pool])


def one_f_one_b_order(
    pp: int, num_micro_batches: int, rank: int, warmup_step: int | list[int] = 1
) -> list[tuple[str, int, int]]:
    """Rank `rank`'s ops in 1F1B, as `(kind, stage, micro_batch)` with kind "F" or "B". Each stage runs
    `warmup_step` more warmup forwards than the next one (an int, or one entry per neighbour pair).

    With the classic step of 1, a stage's gap between F(j) and B(j) is exactly the next stage's work
    on j, so every cycle also waits for one activation and one gradient transfer. A step of 2 gives
    the pair a cycle of slack that hides both transfers, at one more micro-batch in flight on every
    stage before it (`2 * (pp - rank) - 1` on stage `rank` when every step is 2). Mixing them keeps
    memory-bound early stages at step 1, where they then need ~2 transfers less work per cycle than
    the bottleneck."""
    if isinstance(warmup_step, int):
        warmup_step = [warmup_step]
    steps = list(warmup_step) * (pp - 1) if len(warmup_step) == 1 else list(warmup_step)
    if len(steps) != pp - 1 or any(step < 1 for step in steps):
        raise ValueError(f"warmup_step must be one positive step or {pp - 1} of them, got {warmup_step}")
    warmup = min(num_micro_batches, 1 + sum(steps[rank:]))
    order = [("F", rank, mb) for mb in range(warmup)]
    for mb in range(num_micro_batches):
        order.append(("B", rank, mb))
        if warmup + mb < num_micro_batches:
            order.append(("F", rank, warmup + mb))
    return order


def dualpipev_order(pp: int, num_micro_batches: int, rank: int) -> list[tuple[str, int, int]]:
    """Rank `rank`'s ops in DeepSeek's DualPipeV (github.com/deepseek-ai/DualPipe, `dualpipev.py`). The
    rank holds stage `rank` (chunk 0) and stage `2 * pp - 1 - rank` (chunk 1). Every backward is a full
    one (DualPipeV's weight-gradient slots are empty) and each overlapped forward-backward pair runs as
    its forward and then its backward.

    Its early chunk-0 warmup keeps two micro-batches of half-size chunks between neighbours, enough
    slack to hide a transfer per direction per cycle, while no rank holds more than about `pp + 1/2`
    micro-batches of its layers."""
    if num_micro_batches < 2 * pp:
        raise ValueError(f"DualPipeV needs at least {2 * pp} micro-batches, got {num_micro_batches}")
    stage = (rank, 2 * pp - 1 - rank)
    forwards, backwards = [0, 0], [0, 0]
    order = []

    def forward(chunk: int) -> None:
        order.append(("F", stage[chunk], forwards[chunk]))
        forwards[chunk] += 1

    def backward(chunk: int) -> None:
        order.append(("B", stage[chunk], backwards[chunk]))
        backwards[chunk] += 1

    tail = pp - rank - 1
    for _ in range(2 * tail):
        forward(0)
    for _ in range(rank + 1):
        forward(0)
        forward(1)
    for _ in range(tail):
        backward(1)
        forward(1)
    for _ in range(num_micro_batches - 2 * pp + rank + 1):
        forward(0)
        backward(1)
        forward(1)
        backward(0)
    for _ in range(tail):
        backward(1)
        forward(1)
        backward(0)
    for _ in range(rank + 1):
        backward(1)
        backward(0)
    for _ in range(tail):
        backward(0)
    return order


ASYNC_ORDERS = {"Async1F1B": one_f_one_b_order, "DualPipeV": dualpipev_order}
# Ops ahead of the current one whose receives are already posted.
ASYNC_LOOKAHEAD = 1


Op = tuple[str, int, int]  # ("F" | "B", stage, micro_batch)
Action = tuple[str, Op]  # ("recv", consumer op) | ("run", op) | ("send", producer op)


def _transfers(op: Op, owners: dict[int, int], num_stages: int) -> tuple[tuple | None, tuple | None]:
    """The remote transfer `op` consumes and the one it produces, as (payload, src stage, dst stage, mb)."""
    kind, stage, mb = op
    if kind == "F":
        consumed = ("act", stage - 1, stage, mb) if stage > 0 else None
        produced = ("act", stage, stage + 1, mb) if stage < num_stages - 1 else None
    else:
        consumed = ("grad", stage + 1, stage, mb) if stage < num_stages - 1 else None
        produced = ("grad", stage, stage - 1, mb) if stage > 0 else None
    remote = lambda t: t is not None and owners[t[1]] != owners[t[2]]  # noqa: E731
    return (consumed if remote(consumed) else None), (produced if remote(produced) else None)


def pipeline_actions(order: list[Op], owners: dict[int, int], num_stages: int, lookahead: int) -> list[Action]:
    """This rank's actions: each op's receive posted `lookahead` ops early, its send right after it."""
    actions, posted = [], set()
    for pos, op in enumerate(order):
        for ahead in order[pos : pos + 1 + lookahead]:
            if _transfers(ahead, owners, num_stages)[0] is not None and ahead not in posted:
                posted.add(ahead)
                actions.append(("recv", ahead))
        actions.append(("run", op))
        if _transfers(op, owners, num_stages)[1] is not None:
            actions.append(("send", op))
    return actions


def globally_ordered_actions(
    orders: list[list[Op]], owners: dict[int, int], num_stages: int, rank: int, lookahead: int
) -> list[Action]:
    """Like `pipeline_actions`, but every rank posts its transfers in one global order, the order in
    which a nominal run (forward 1, backward 2 per stage) produces them.

    Collectives that NCCL runs on copy engines (`CopyEngineEdge`) complete in issue order across all of
    a device's communicators: two ranks issuing two such transfers in opposite orders deadlock.
    Ranks posting a shared transfer at the same place in one total order cannot, even when each
    transfer is completed before going on (how `AsyncPipelineSchedule` runs its first step). A receive an op needs
    is always postable before it (whatever precedes it in the order was produced earlier in the nominal
    run, hence by ops already issued here); receives further ahead are posted when the order allows."""
    end: dict[Op, float] = {}
    pos = [0] * len(orders)
    free = [0.0] * len(orders)
    arrive: dict[tuple, float] = {}
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for r, ops in enumerate(orders):
            while pos[r] < len(ops):
                op = ops[pos[r]]
                kind, stage, mb = op
                if kind == "F":
                    dep = 0.0 if stage == 0 else arrive.get(("act", stage - 1, stage, mb))
                else:
                    dep = end.get(("F", stage, mb))
                    if dep is not None and stage < num_stages - 1:
                        grad = arrive.get(("grad", stage + 1, stage, mb))
                        dep = None if grad is None else max(dep, grad)
                if dep is None:
                    break
                start = max(dep, free[r])
                end[op] = free[r] = start + (1.0 if kind == "F" else 2.0)
                if kind == "F" and stage < num_stages - 1:
                    arrive[("act", stage, stage + 1, mb)] = end[op]
                if kind == "B" and stage > 0:
                    arrive[("grad", stage, stage - 1, mb)] = end[op]
                pos[r] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("pipeline orders deadlock")
    producer = {}
    for ops in orders:
        for op in ops:
            produced = _transfers(op, owners, num_stages)[1]
            if produced is not None:
                producer[produced] = op
    mine = sorted(
        (t for t in producer if rank in (owners[t[1]], owners[t[2]])),
        key=lambda t: (end[producer[t]], t[1], t[2], t[3], t[0]),
    )
    consumer = {}
    for op in orders[rank]:
        consumed = _transfers(op, owners, num_stages)[0]
        if consumed is not None:
            consumer[consumed] = op
    index = {t: i for i, t in enumerate(mine)}
    actions, done, nxt = [], set(), 0

    def flush(upto: int, required: bool) -> None:
        nonlocal nxt
        while nxt <= upto:
            t = mine[nxt]
            if owners[t[1]] == rank:
                if producer[t] not in done:
                    if required:
                        raise RuntimeError(f"transfer {t} is needed before its producer runs")
                    return
                actions.append(("send", producer[t]))
            else:
                actions.append(("recv", consumer[t]))
            nxt += 1

    order = orders[rank]
    for i, op in enumerate(order):
        consumed = _transfers(op, owners, num_stages)[0]
        if consumed is not None:
            flush(index[consumed], required=True)
        for ahead in order[i + 1 : i + 1 + lookahead]:
            consumed = _transfers(ahead, owners, num_stages)[0]
            if consumed is not None:
                flush(index[consumed], required=False)
        actions.append(("run", op))
        done.add(op)
        produced = _transfers(op, owners, num_stages)[1]
        if produced is not None:
            flush(index[produced], required=False)
    flush(len(mine) - 1, required=True)
    return actions


def action_pool_sizes(actions: list[Action]) -> dict[int, int]:
    """Receive buffer sets each stage needs: an activation buffer lives from its receive's post until
    its micro-batch's backward runs."""
    live, peak = {}, {}
    for action, (kind, stage, _) in actions:
        live.setdefault(stage, 0)
        if action == "recv" and kind == "F":
            live[stage] += 1
        if action == "run" and kind == "B":
            live[stage] -= 1
        peak[stage] = max(peak.get(stage, 0), live[stage])
    return {stage: n + 1 for stage, n in peak.items()}


def pipeline_edge_groups(
    pp_ranks: list[int],
    pp_rank: int,
    kinds: int,
    copy_engine: bool = False,
    ctas: int | None = None,
    device: torch.device | str = "cuda",
) -> dict[tuple[int, int], dist.ProcessGroup]:
    """`kinds` two-rank NCCL groups per pair of neighbouring pipeline ranks, keyed `(lower rank, kind)`.

    Each transfer edge of the pipeline gets its own communicator, so its traffic runs on its own stream
    and both ends post it in micro-batch order. Every rank creates the same number of groups in the same
    order (a singleton where it has no neighbour), as locally synchronized groups need, and the
    communicators are connected in one global order so their lazy initialization cannot deadlock.
    `copy_engine` gives the groups NCCL's zero-CTA policy (see `CopyEngineEdge` and `PutEdge`); `ctas` pins the number
    of CTAs (SMs) a send/recv kernel takes."""
    options = None
    if copy_engine or ctas is not None:
        options = dist.ProcessGroupNCCL.Options()
    if copy_engine:
        options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    elif ctas is not None:
        options.config.min_ctas = options.config.max_ctas = ctas
    groups, mine = {}, []
    for kind in range(kinds):
        for parity in (0, 1):
            low = pp_rank if pp_rank % 2 == parity else pp_rank - 1
            if 0 <= low and low + 1 < len(pp_ranks):
                group = dist.new_group(
                    [pp_ranks[low], pp_ranks[low + 1]], use_local_synchronization=True, pg_options=options
                )
                groups[(low, kind)] = group
                mine.append((group, pp_ranks[low + 1] if low == pp_rank else pp_ranks[low]))
            else:
                dist.new_group([pp_ranks[pp_rank]], use_local_synchronization=True)
    for group, peer in mine:
        send, recv = torch.zeros(1, device=device), torch.empty(1, device=device)
        ops = [dist.P2POp(dist.isend, send, peer, group), dist.P2POp(dist.irecv, recv, peer, group)]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return groups


class _EventWork:
    """A transfer that is done once `event` (recorded on a side stream) is."""

    def __init__(self, event: torch.cuda.Event):
        self.event = event

    def wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.event)

    def is_completed(self) -> bool:
        return self.event.query()


class CopyEngineEdge:
    """One transfer edge carried by an in-place two-rank all-gather of a buffer registered as an NCCL
    symmetric window, on a group with the zero-CTA policy. NCCL >= 2.32 runs that all-gather on copy
    engines, so a transfer in flight takes no SM from the compute it overlaps (a send/recv kernel would
    hold SMs and slow persistent kernels such as the fused MoE and DeepGEMM by up to ~2x).

    The sender packs its tensors into its half of the buffer; the receiver's half goes the other way
    unread (all-gathers have equal parts). Packing and unpacking are device copies on the edge's own
    stream. The buffer is reused by the next transfer on the edge once this one is done."""

    ALIGN = 256

    def __init__(self, group: dist.ProcessGroup, nbytes: int, device: torch.device):
        backend = group._get_backend(device)
        self.part = -(-nbytes // self.ALIGN) * self.ALIGN
        self.pool = torch.cuda.MemPool(backend.mem_allocator)
        with torch.cuda.use_mem_pool(self.pool):
            self.buffer = torch.zeros(2 * self.part, dtype=torch.uint8, device=device)
        backend.register_mem_pool(self.pool, symm=True)
        self.group = group
        self.index = dist.get_group_rank(group, dist.get_rank())
        self.stream = torch.cuda.Stream(device)
        self.last: dist.Work | None = None

    def _views(self, half: int, tensors: list[Tensor]) -> list[Tensor]:
        views, offset = [], half * self.part
        for tensor in tensors:
            nbytes = tensor.numel() * tensor.element_size()
            views.append(self.buffer[offset : offset + nbytes].view(tensor.dtype).view(tensor.shape))
            offset += -(-nbytes // self.ALIGN) * self.ALIGN
        if offset > (half + 1) * self.part:
            raise ValueError(f"stage transfer of {offset - half * self.part} bytes exceeds its {self.part}-byte buffer")
        return views

    def _all_gather(self) -> dist.Work:
        mine = self.buffer[self.index * self.part : (self.index + 1) * self.part]
        self.last = dist.all_gather_into_tensor(self.buffer, mine, group=self.group, async_op=True)
        return self.last

    def send(self, tensors: list[Tensor]) -> list[dist.Work]:
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            if self.last is not None:
                self.last.wait()
            for tensor, view in zip(tensors, self._views(self.index, tensors)):
                tensor.record_stream(self.stream)
                view.copy_(tensor)
            return [self._all_gather()]

    def recv(self, tensors: list[Tensor]) -> list[_EventWork]:
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self._all_gather().wait()
            for tensor, view in zip(tensors, self._views(1 - self.index, tensors)):
                # Receive buffers double as autograd leaves of the stage's input.
                tensor.detach().copy_(view)
            return [_EventWork(self.stream.record_event())]


class _WaitSignalDesc(ctypes.Structure):
    """NCCL's `ncclWaitSignalDesc_t`."""

    _fields_ = [("opCnt", ctypes.c_int), ("peer", ctypes.c_int), ("sigIdx", ctypes.c_int), ("ctx", ctypes.c_int)]


_NCCL_UINT8 = 1
_NCCL_WIN_COLL_SYMMETRIC = 1
_NCCL_MIN_PUT_VERSION = 23000


def _nccl_lib() -> ctypes.CDLL:
    """The libnccl torch loaded (dlopen returns the already loaded library, also when one is preloaded)."""
    lib = ctypes.CDLL("libnccl.so.2")
    version = ctypes.c_int()
    _nccl_check(lib, lib.ncclGetVersion(ctypes.byref(version)), "ncclGetVersion")
    if version.value < _NCCL_MIN_PUT_VERSION:
        raise RuntimeError(f"pp_transport='put' needs NCCL >= 2.30, the loaded NCCL is {version.value}")
    return lib


def _nccl_check(lib: ctypes.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(f"{what} failed: {lib.ncclGetErrorString(rc).decode()} ({rc})")


class PutEdge:
    """One transfer edge (one direction between two neighbouring ranks) carried by NCCL's one-sided
    `ncclPutSignal` on a two-rank group with the zero-CTA policy, which NCCL >= 2.30 drives from a CPU
    proxy: a transfer in flight holds no SM.

    Both ends register a window of one micro-batch: the sender's is its pack buffer (a put's source must
    be in a window), the receiver's is the landing buffer. The sender packs the micro-batch and puts it
    with a signal; the receiver waits for the signal, copies the micro-batch out and signals back a credit,
    which the sender waits for before its next put. The group carries one direction only, so a signal from
    the peer is unambiguous. Puts on different edges need no common order (unlike `CopyEngineEdge`)."""

    ALIGN = 256

    def __init__(self, group: dist.ProcessGroup, nbytes: int, device: torch.device):
        backend = group._get_backend(device)
        self.lib = _nccl_lib()
        self.part = -(-nbytes // self.ALIGN) * self.ALIGN
        self.pool = torch.cuda.MemPool(backend.mem_allocator)
        with torch.cuda.use_mem_pool(self.pool):
            self.buffer = torch.zeros(self.part, dtype=torch.uint8, device=device)
        torch.cuda.synchronize()
        self.comm = ctypes.c_void_p(backend._comm_ptr())
        self.window = ctypes.c_void_p()
        _nccl_check(
            self.lib,
            self.lib.ncclCommWindowRegister(
                self.comm,
                ctypes.c_void_p(self.buffer.data_ptr()),
                ctypes.c_size_t(self.part),
                ctypes.byref(self.window),
                ctypes.c_int(_NCCL_WIN_COLL_SYMMETRIC),
            ),
            "ncclCommWindowRegister",
        )
        self.peer = 1 - dist.get_group_rank(group, dist.get_rank())
        self.stream = torch.cuda.Stream(device)
        self.sent = 0

    def _views(self, tensors: list[Tensor]) -> tuple[list[Tensor], int]:
        """Views of the buffer for `tensors`, and the bytes up to the end of the last one."""
        views, offset, end = [], 0, 0
        for tensor in tensors:
            nbytes = tensor.numel() * tensor.element_size()
            views.append(self.buffer[offset : offset + nbytes].view(tensor.dtype).view(tensor.shape))
            end = offset + nbytes
            offset += -(-nbytes // self.ALIGN) * self.ALIGN
        if end > self.part:
            raise ValueError(f"stage transfer of {end} bytes exceeds its {self.part}-byte buffer")
        return views, end

    def _wait_signal(self) -> None:
        desc = _WaitSignalDesc(1, self.peer, 0, 0)
        stream = ctypes.c_void_p(self.stream.cuda_stream)
        _nccl_check(
            self.lib, self.lib.ncclWaitSignal(ctypes.c_int(1), ctypes.byref(desc), self.comm, stream), "ncclWaitSignal"
        )

    def send(self, tensors: list[Tensor]) -> list[_EventWork]:
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            if self.sent:
                self._wait_signal()  # the receiver has copied the previous micro-batch out
            views, nbytes = self._views(tensors)
            for tensor, view in zip(tensors, views):
                tensor.record_stream(self.stream)
                view.copy_(tensor)
            _nccl_check(
                self.lib,
                self.lib.ncclPutSignal(
                    ctypes.c_void_p(self.buffer.data_ptr()),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(_NCCL_UINT8),
                    ctypes.c_int(self.peer),
                    self.window,
                    ctypes.c_size_t(0),  # offset in the peer's window
                    ctypes.c_int(0),  # signal index
                    ctypes.c_int(0),  # context
                    ctypes.c_uint(0),  # flags
                    self.comm,
                    ctypes.c_void_p(self.stream.cuda_stream),
                ),
                "ncclPutSignal",
            )
            self.sent += 1
            return [_EventWork(self.stream.record_event())]

    def recv(self, tensors: list[Tensor]) -> list[_EventWork]:
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self._wait_signal()
            for tensor, view in zip(tensors, self._views(tensors)[0]):
                # Receive buffers double as autograd leaves of the stage's input.
                tensor.detach().copy_(view)
            _nccl_check(
                self.lib,
                self.lib.ncclSignal(
                    ctypes.c_int(self.peer),
                    ctypes.c_int(0),  # signal index
                    ctypes.c_int(0),  # context
                    ctypes.c_uint(0),  # flags
                    self.comm,
                    ctypes.c_void_p(self.stream.cuda_stream),
                ),
                "ncclSignal",
            )
            return [_EventWork(self.stream.record_event())]


def _release_outputs(stage: PipelineStage, mb: int) -> None:
    """Free the memory of a micro-batch's sent forward outputs that the model declares releasable
    (`pipeline_releasable_outputs`, output indices): outputs that no backward of the stage reads, so the
    backward only needs their autograd graph (as Megatron's `deallocate_output_tensor`). An output keeps
    its shape over a one-element storage, so the shape checks of the backward still pass. Any other
    output may be saved for the backward of a later op in the stage and is kept."""
    releasable = getattr(stage.submod, "pipeline_releasable_outputs", ())
    if not releasable or mb not in stage.fwd_cache:  # nothing to free, or its backward already ran
        return
    outputs, _ = stage.fwd_cache[mb]
    for idx in releasable:
        out = outputs[idx]
        if isinstance(out, Tensor) and out.grad_fn is not None:
            out.data = torch.empty(1, dtype=out.dtype, device=out.device).expand(out.shape)


class AsyncPipelineSchedule(PipelineScheduleMulti):
    """Runs a static per-rank list of actions (`pipeline_actions`, `globally_ordered_actions`): post an
    op's receive, run an op, post the send it produced.

    torch's schedules post each receive just before the compute that reads it and wait on the sends,
    which puts every transfer on the critical path. Here receives are posted ahead (their transfer can
    start as soon as the compute issued before them ends), the compute stream waits on a receive only
    right before reading it, and sends are never waited on before later compute. Each transfer edge has
    its own communicator (`pipeline_edge_groups`), so traffic in flight on one edge never holds up
    another. Stages on the same rank hand tensors over directly."""

    def __init__(
        self,
        stages: list[PipelineStage],
        *,
        actions: list[Action],
        owners: dict[int, int],
        edge_groups: dict[tuple[int, int], dist.ProcessGroup],
        edge_kind: Callable[[int, int], int],
        copy_engine_edges: dict[tuple[int, int], CopyEngineEdge | PutEdge] | None = None,
        first_step_actions: list[Action] | None = None,
        offload: PipelineActivationOffloadConfig | None = None,
        defer_expert_weight_grads: bool = False,
        **kwargs,
    ):
        super().__init__(stages, **kwargs)
        self._defer_expert_weight_grads = defer_expert_weight_grads
        self._offloader = None
        if offload is not None:
            self._offloader = PipelineActivationOffloader(
                [stage.submod for stage in stages], offload.min_bytes, offload.target
            )
            run_ops = [op for action, op in actions if action == "run"]
            stages_set = None if offload.stages is None else set(offload.stages)
            self._offloaded, self._prefetch_at = offload_plan(run_ops, stages_set, offload.prefetch_ahead)
        self.stage_index_to_group_rank = dict(owners)
        for stage in stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank
        self._actions, self._owners = actions, owners
        self._first_step_actions = first_step_actions
        self._edge_groups, self._edge_kind = edge_groups, edge_kind
        self._copy_engine_edges = copy_engine_edges
        self._rank = owners[stages[0].stage_index]

    def _post(self, ops: list[dist.P2POp], src: int, dst: int) -> list[dist.Work]:
        """Post the transfer of edge `src` -> `dst` (stages on neighbouring ranks) on its own group."""
        if not ops:
            return []
        low = min(self._owners[src], self._owners[dst])
        key = (low, self._edge_kind(src, dst))
        if self._copy_engine_edges is not None:
            edge = self._copy_engine_edges[key]
            tensors = [op.tensor for op in ops]
            return edge.send(tensors) if ops[0].op is dist.isend else edge.recv(tensors)
        group = self._edge_groups[key]
        return dist.batch_isend_irecv([dist.P2POp(op.op, op.tensor, op.peer, group) for op in ops])

    def _step_microbatches(
        self, arg_mbs=None, kwarg_mbs=None, target_mbs=None, losses=None, return_outputs=True, loss_kwargs=None
    ):
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stages(arg_mbs[0], kwarg_mbs[0], first_target, loss_kwargs)
        stages = {stage.stage_index: stage for stage in self._stages}
        n = self._n_microbatches
        recvs: dict[Op, list[dist.Work]] = {}
        sends: list[dist.Work] = []
        # Forward outputs sent to another rank, released once their transfer is done (see `_release_output`).
        sent_outputs: list[tuple[list[dist.Work], PipelineStage, int]] = []
        # The first step runs every transfer to completion, in one global order, before going on: it
        # is when Triton autotunes the backward kernels, and the autotuner synchronizes the device,
        # which would wait forever on a transfer posted ahead whose peer is synchronizing too.
        blocking = self._first_step_actions is not None
        actions = self._first_step_actions if blocking else self._actions
        self._first_step_actions = None
        # The first step keeps every activation on the GPU: its global action order differs from the plan's.
        offloader = None if blocking else self._offloader
        # Expert weight-gradient GEMMs of the last backward, run once its input-gradient send is posted.
        weight_grads: list[Callable[[], None]] = []

        def run_weight_grads() -> None:
            for weight_grad in weight_grads:
                weight_grad()
            weight_grads.clear()

        run_index = 0
        for action, op in actions:
            kind, idx, mb = op
            stage = stages[idx]
            if action == "run":
                run_weight_grads()
            if action == "recv":
                if kind == "F":
                    recvs[op] = self._post(stage.get_fwd_recv_ops(mb), idx - 1, idx)
                else:
                    recvs[op] = self._post(stage.get_bwd_recv_ops(mb), idx + 1, idx)
                if blocking:
                    for work in recvs[op]:
                        work.wait()
                    torch.cuda.synchronize()
            elif action == "send":
                if kind == "F":
                    works = self._post(stage.get_fwd_send_ops(mb), idx, idx + 1)
                    sends.extend(works)
                    sent_outputs.append((works, stage, mb))
                else:
                    sends.extend(self._post(stage.get_bwd_send_ops(mb), idx, idx - 1))
                    run_weight_grads()
                if blocking:
                    for work in sends:
                        work.wait()
                    torch.cuda.synchronize()
                # Completed sends release their tensors (input gradients are not referenced elsewhere).
                sends = [work for work in sends if not work.is_completed()]
                pending = []
                for works, sent_stage, sent_mb in sent_outputs:
                    if all(work.is_completed() for work in works):
                        _release_outputs(sent_stage, sent_mb)
                    else:
                        pending.append((works, sent_stage, sent_mb))
                sent_outputs = pending
            else:
                for work in recvs.pop(op, []):
                    work.wait()
                if offloader is not None:
                    for key in self._prefetch_at.get(run_index, ()):
                        offloader.prefetch(key)
                run_index += 1
                offload = offloader is not None and (idx, mb) in self._offloaded
                with torch.profiler.record_function(f"pp.{kind}.stage{idx}.mb{mb}"):
                    if kind == "F":
                        with offloader.forward((idx, mb)) if offload else nullcontext():
                            output = stage.forward_one_chunk(
                                mb, arg_mbs[mb], kwarg_mbs[mb], save_forward_output=return_outputs
                            )
                        if offload:
                            offloader.swap_out((idx, mb), output)
                        self._maybe_compute_loss(stage, output, target_mbs, mb, loss_kwargs)
                        if not stage.is_last and idx + 1 in stages:
                            stages[idx + 1].set_local_fwd_input(output, mb)
                    else:
                        loss = self._maybe_get_loss(stage, mb)
                        if offload:
                            offloader.wait((idx, mb))
                        defer = self._defer_expert_weight_grads and not blocking and mb != n - 1
                        with defer_weight_grads() if defer else nullcontext([]) as deferred:
                            stage.backward_one_chunk(mb, loss=loss, last_backward=mb == n - 1)
                            weight_grads.extend(deferred)
                            deferred.clear()
                        if not stage.is_first and idx - 1 in stages:
                            stages[idx - 1].set_local_bwd_input(stage.get_local_bwd_output(mb), mb)
        run_weight_grads()
        for work in sends:
            work.wait()
        self._update_losses(self._stages, losses)
        for stage in self._stages:
            stage.perform_reduce_grad(n if self.scale_grads else 1)


def num_pipeline_stages(pp: int, stages_per_rank: int) -> int:
    return pp * stages_per_rank


def local_stage_ids(pp_rank: int, pp: int, schedule: str, stages_per_rank: int) -> list[int]:
    """The stages this pipeline rank holds, in increasing order."""
    if schedule in SINGLE_STAGE_SCHEDULES or schedule == "Async1F1B":
        if stages_per_rank != 1:
            raise ValueError(f"the {schedule} schedule runs one stage per rank, got {stages_per_rank}")
        return [pp_rank]
    if schedule in V_SCHEDULES:
        if stages_per_rank != 2:
            raise ValueError(f"the {schedule} schedule runs two stages per rank, got {stages_per_rank}")
        return [pp_rank, 2 * pp - 1 - pp_rank]
    return [pp_rank + i * pp for i in range(stages_per_rank)]


def stage_layer_units(
    num_layers: int, num_stages: int, stage: int, layers_per_stage: list[float] | None = None
) -> range:
    """Consecutive half-layer units of `stage`: unit 2i is layer i's attention block and 2i + 1 its MoE
    block. `layers_per_stage` may cut a layer between the two (multiples of 0.5); by default whole
    layers are split as evenly as possible, earlier stages taking the remainder."""
    if layers_per_stage is None:
        layers_per_stage = [
            num_layers // num_stages + (1 if i < num_layers % num_stages else 0) for i in range(num_stages)
        ]
    units = [round(2 * layers) for layers in layers_per_stage]
    if (
        len(units) != num_stages
        or sum(units) != 2 * num_layers
        or any(u < 0 or u != 2 * layers for u, layers in zip(units, layers_per_stage))
    ):
        raise ValueError(
            f"layers_per_stage {layers_per_stage} must give {num_stages} stages {num_layers} layers in halves"
        )
    start = sum(units[:stage])
    return range(start, start + units[stage])


def prune_to_pipeline_stage(
    model: nn.Module, stage: int, num_stages: int, layers_per_stage: list[float] | None = None
) -> None:
    if not hasattr(model, "pipeline_stage_forward"):
        raise ValueError(f"{type(model).__name__} does not support pipeline parallelism")
    units = stage_layer_units(len(model.model.layers), num_stages, stage, layers_per_stage)
    model.prune_to_pipeline_stage(units, first=stage == 0, last=stage == num_stages - 1)
    model.forward = model.pipeline_stage_forward
    # For per-stage policies applied after pruning (e.g. activation checkpointing).
    model.pipeline_stage, model.pipeline_num_stages = stage, num_stages


def to_meta(tensors: tuple[Tensor, ...], differentiable: bool) -> tuple[Tensor, ...]:
    # The probe ran without grad; every floating-point tensor between stages may carry one.
    return tuple(
        torch.empty(t.shape, dtype=t.dtype, device="meta").requires_grad_(differentiable and t.is_floating_point())
        for t in tensors
    )


@torch.no_grad()
def probe_stage_shapes(
    model_parts: list[nn.Module],
    stage_ids: list[int],
    num_stages: int,
    parallel_dims: ParallelDims,
    first_inputs: tuple[Tensor, ...] | None,
    owners: dict[int, int],
) -> list[tuple[tuple[Tensor, ...], tuple[Tensor, ...]]]:
    """Each local stage's input and output tensors on one micro-batch, as meta tensors, from a
    no-grad forward through every stage in order. Stages exchange fixed shapes, and the pipeline's
    own shape inference would run every later stage on uninitialized inputs."""
    pp_mesh = parallel_dims.world_mesh["pp"]
    group, rank = pp_mesh.get_group(), pp_mesh.get_local_rank()
    shapes, carried = [], None
    for stage in range(num_stages):
        if stage not in stage_ids:
            continue
        if stage == 0:
            inputs = first_inputs
        elif owners[stage - 1] == rank:
            inputs = carried
        else:
            metas = [None]
            dist.recv_object_list(metas, group_src=owners[stage - 1], group=group)
            inputs = tuple(torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in metas[0])
            for tensor in inputs:
                dist.recv(tensor, group_src=owners[stage - 1], group=group)
        model = model_parts[stage_ids.index(stage)]
        outputs = model(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        carried = outputs
        if stage < num_stages - 1 and owners[stage + 1] != rank:
            dst = owners[stage + 1]
            dist.send_object_list([[(t.shape, t.dtype) for t in outputs]], group_dst=dst, group=group)
            for tensor in outputs:
                dist.send(tensor.contiguous(), group_dst=dst, group=group)
        # The head's FSDP group opts out of resharding after forward; nothing else may stay gathered.
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        shapes.append((to_meta(inputs, differentiable=stage > 0), to_meta(outputs, differentiable=True)))
    return shapes


def build_pipeline_schedule(
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
    schedule: str,
    stages_per_rank: int,
    num_micro_batches: int,
    loss_fn: Callable[[Tensor, Tensor | None], Tensor],
    first_inputs: tuple[Tensor, ...] | None,
    warmup_step: int | list[int] = 1,
    transport: str = "nccl",
    transport_ctas: int | None = None,
    offload: PipelineActivationOffloadConfig | None = None,
    defer_expert_weight_grads: bool = False,
) -> PipelineScheduleSingle | PipelineScheduleMulti:
    """`first_inputs` is stage 0's first micro-batch (`None` on ranks without stage 0)."""
    if "ZeroBubble" in schedule:
        # Zero-bubble schedules split each backward into an input-gradient and a weight-gradient
        # pass over a retained graph, which compiled blocks with donated buffers refuse.
        torch._functorch.config.donated_buffer = False
    pp_mesh = parallel_dims.world_mesh["pp"]
    pp = parallel_dims.pp
    num_stages = num_pipeline_stages(pp, stages_per_rank)
    stage_ids = local_stage_ids(pp_mesh.get_local_rank(), pp, schedule, stages_per_rank)
    owners = {stage: owner for owner in range(pp) for stage in local_stage_ids(owner, pp, schedule, stages_per_rank)}
    shapes = probe_stage_shapes(model_parts, stage_ids, num_stages, parallel_dims, first_inputs, owners)
    device = torch.device("cuda", torch.cuda.current_device())
    if schedule in ASYNC_ORDERS:
        rank = pp_mesh.get_local_rank()

        def rank_order(r: int) -> list[Op]:
            if schedule == "Async1F1B":
                return one_f_one_b_order(pp, num_micro_batches, r, warmup_step)
            return ASYNC_ORDERS[schedule](pp, num_micro_batches, r)

        orders = [rank_order(r) for r in range(pp)]
        global_actions = globally_ordered_actions(orders, owners, num_stages, rank, ASYNC_LOOKAHEAD)
        if transport == "copy_engine":
            actions = global_actions
        else:
            actions = pipeline_actions(orders[rank], owners, num_stages, ASYNC_LOOKAHEAD)
        pools = action_pool_sizes(actions)
        for stage, size in action_pool_sizes(global_actions).items():
            pools[stage] = max(pools.get(stage, 0), size)

        # Stages fed by a stage on the same rank take its tensors directly and need one buffer set.
        def local_input(stage: int, step: int) -> bool:
            return stage + step in owners and owners[stage + step] == owners[stage]

        stages = [
            PooledRecvPipelineStage(
                model,
                stage_index=stage,
                num_stages=num_stages,
                device=device,
                input_args=input_args,
                output_args=output_args,
                group=pp_mesh.get_group(),
                pool=1 if local_input(stage, -1) else pools[stage],
                grad_pool=1 if local_input(stage, 1) else ASYNC_LOOKAHEAD + 1,
            )
            for model, stage, (input_args, output_args) in zip(model_parts, stage_ids, shapes)
        ]

        # Edge kinds per pair of neighbouring ranks: (chunk, activation or gradient).
        def edge_kind(src: int, dst: int) -> int:
            return 2 * (max(src, dst) >= pp) + (dst < src)

        pp_ranks = dist.get_process_group_ranks(pp_mesh.get_group())
        edge_groups = pipeline_edge_groups(
            pp_ranks,
            pp_mesh.get_local_rank(),
            kinds=2 * stages_per_rank,
            copy_engine=transport in ("copy_engine", "put"),
            ctas=transport_ctas,
        )
        copy_engine_edges = None
        if transport in ("copy_engine", "put"):
            edge_cls = CopyEngineEdge if transport == "copy_engine" else PutEdge
            # An edge's buffer holds one micro-batch of the activations crossing it (its gradients are
            # no larger). Registration is collective, so edges are set up in the groups' global order.
            local = dict(zip(stage_ids, shapes))
            copy_engine_edges = {}
            for low, kind in sorted(edge_groups, key=lambda key: (key[1], key[0] % 2)):
                first = low if kind < 2 else 2 * pp - 2 - low
                metas = local[first][1] if first in local else local[first + 1][0]
                nbytes = sum(meta.numel() * meta.element_size() for meta in metas)
                copy_engine_edges[(low, kind)] = edge_cls(edge_groups[(low, kind)], nbytes, device)
            torch.cuda.synchronize()
        # Gradients are scaled by the caller, like the gradient-accumulation path does.
        return AsyncPipelineSchedule(
            stages,
            actions=actions,
            owners=owners,
            edge_groups=edge_groups,
            edge_kind=edge_kind,
            copy_engine_edges=copy_engine_edges,
            first_step_actions=global_actions,
            offload=offload,
            defer_expert_weight_grads=defer_expert_weight_grads,
            n_microbatches=num_micro_batches,
            loss_fn=loss_fn,
            scale_grads=False,
        )
    # 1F1B keeps at most `pp` micro-batches per stage in flight; two spare sets cover receives
    # posted next to the previous micro-batch's backward. The multi-stage schedules keep at most
    # one per stage of the pipeline. GPipe keeps them all.
    pool = {"1F1B": pp + 2, "GPipe": None}.get(schedule, 2 * num_stages + 2)
    stage_cls = partial(PooledRecvPipelineStage, pool=pool) if pool is not None else PipelineStage
    stages = [
        stage_cls(
            model,
            stage_index=stage,
            num_stages=num_stages,
            device=device,
            input_args=input_args,
            output_args=output_args,
            group=pp_mesh.get_group(),
        )
        for model, stage, (input_args, output_args) in zip(model_parts, stage_ids, shapes)
    ]
    # Gradients are scaled by the caller, like the gradient-accumulation path does.
    if schedule in SINGLE_STAGE_SCHEDULES:
        return SINGLE_STAGE_SCHEDULES[schedule](
            stages[0], n_microbatches=num_micro_batches, loss_fn=loss_fn, scale_grads=False
        )
    return MULTI_STAGE_SCHEDULES[schedule](stages, n_microbatches=num_micro_batches, loss_fn=loss_fn, scale_grads=False)


def queue_seq_lens(model_parts: list[nn.Module], micro_batches: list[dict]) -> None:
    """Hand every stage the host-side document lengths of the micro-batches it is about to run."""
    for part in model_parts:
        part.pipeline_seq_lens = [mb["seq_lens"].flatten().cpu() for mb in micro_batches]


def stack_micro_batches(micro_batches: list[dict]) -> tuple[Tensor, ...]:
    """The step's micro-batches stacked along the batch dim, as the first stage takes them:
    `(input_ids, position_ids, labels)`. Checks on the host that positions restart at 0 at every
    document start."""
    for mb in micro_batches:
        seq_lens = mb["seq_lens"].flatten()
        starts = torch.cumsum(seq_lens, 0) - seq_lens
        if (mb["position_ids"].flatten()[starts] != 0).any():
            raise ValueError("position_ids must restart at 0 at every document boundary of seq_lens")
    input_ids = torch.cat([mb["input_ids"] for mb in micro_batches])
    position_ids = torch.cat([mb["position_ids"] for mb in micro_batches])
    labels = torch.cat([mb["target_ids"].masked_fill(~mb["loss_mask"], IGNORE_INDEX) for mb in micro_batches])
    return tuple(t.to("cuda", non_blocking=True) for t in (input_ids, position_ids, labels))
