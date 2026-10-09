"""Pipeline parallelism: the decoder layers are split into stages of consecutive layers, and each
pipeline rank holds one or more stages (one model part per stage).

A model opts in with `prune_to_pipeline_stage(layer_ids, first=..., last=...)`, which drops what
the stage does not hold, and `pipeline_stage_forward(*tensors)`, which runs the stage on one
micro-batch: the first stage takes `(input_ids, position_ids, labels)` and the last one returns
the summed loss. Stages are built before FSDP and expert parallelism, which then apply to each
stage over its own ranks.
"""

from collections.abc import Callable
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

from prime_rl.trainer.models.layers.lm_head import IGNORE_INDEX
from prime_rl.trainer.parallel_dims import ParallelDims

SINGLE_STAGE_SCHEDULES: dict[str, type[PipelineScheduleSingle]] = {"1F1B": Schedule1F1B, "GPipe": ScheduleGPipe}
MULTI_STAGE_SCHEDULES: dict[str, type[PipelineScheduleMulti]] = {
    "Interleaved1F1B": ScheduleInterleaved1F1B,
    "InterleavedZeroBubble": ScheduleInterleavedZeroBubble,
    "ZBVZeroBubble": ScheduleZBVZeroBubble,
}
# Schedules whose stages zig-zag over the ranks: rank r holds stages r and 2 * pp - 1 - r.
V_SCHEDULES = {"ZBVZeroBubble"}


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
    """A `PipelineStage` whose per-micro-batch receive buffers (activations and their gradients)
    cycle through `pool` sets instead of one set per micro-batch, so memory does not grow with
    the number of micro-batches. Only valid for schedules that keep fewer than `pool` micro-batches
    of a stage between their forward and backward (1F1B keeps at most the number of stages)."""

    def __init__(self, *args, pool: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = pool

    def _setup_forward_recv_info(self, num_microbatches: int, has_backward: bool) -> None:
        super()._setup_forward_recv_info(num_microbatches, has_backward)
        self._share_buffers(self.args_recv_info, num_microbatches)

    def _setup_backward_recv_info(self, num_microbatches: int) -> None:
        super()._setup_backward_recv_info(num_microbatches)
        self._share_buffers(self.grad_recv_info, num_microbatches)

    def _share_buffers(self, recv_infos: dict, num_microbatches: int) -> None:
        for chunk in range(self.pool, num_microbatches):
            for info, shared in zip(recv_infos[chunk], recv_infos[chunk % self.pool]):
                if info.buffer is not None:
                    info.buffer = shared.buffer


class AsyncSchedule1F1B(Schedule1F1B):
    """1F1B whose sends overlap the following compute.

    torch's 1F1B fuses each send with the next receive and waits on both, so the activation or
    gradient transfer sits on the critical path and adjacent stages run in lockstep. Here
    activations travel on one copy of the pipeline group and gradients on another; each group
    then carries traffic in one direction only, so a send left in flight cannot hold up a
    receive. Receives are waited before the compute that reads them, sends at the end of the step.
    """

    def __init__(self, stage: PipelineStage, *, fwd_group: dist.ProcessGroup, bwd_group: dist.ProcessGroup, **kwargs):
        super().__init__(stage, **kwargs)
        self._fwd_group, self._bwd_group = fwd_group, bwd_group

    @staticmethod
    def _on(ops: list[dist.P2POp], group: dist.ProcessGroup) -> list[dist.Work]:
        if not ops:
            return []
        return dist.batch_isend_irecv([dist.P2POp(op.op, op.tensor, op.peer, group) for op in ops])

    def _step_microbatches(
        self, arg_mbs=None, kwarg_mbs=None, target_mbs=None, losses=None, return_outputs=True, loss_kwargs=None
    ):
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0], first_target, loss_kwargs)
        stage, n = self._stage, self._n_microbatches
        sends: list[dist.Work] = []

        def forward(mb: int) -> None:
            for work in self._on(stage.get_fwd_recv_ops(mb), self._fwd_group):
                work.wait()
            output = stage.forward_one_chunk(mb, arg_mbs[mb], kwarg_mbs[mb], save_forward_output=return_outputs)
            sends.extend(self._on(stage.get_fwd_send_ops(mb), self._fwd_group))
            self._maybe_compute_loss(stage, output, target_mbs, mb, loss_kwargs)

        def backward(mb: int) -> None:
            for work in self._on(stage.get_bwd_recv_ops(mb), self._bwd_group):
                work.wait()
            stage.backward_one_chunk(mb, loss=self._maybe_get_loss(stage, mb), last_backward=mb == n - 1)
            sends.extend(self._on(stage.get_bwd_send_ops(mb), self._bwd_group))

        warmup = min(n, self._num_stages - stage.stage_index)
        for mb in range(warmup):
            forward(mb)
        for mb in range(n):
            backward(mb)
            if warmup + mb < n:
                forward(warmup + mb)
        for work in sends:
            work.wait()
        self._update_losses(stage, losses)
        stage.perform_reduce_grad(n if self.scale_grads else 1)


def directional_pipeline_groups(parallel_dims: ParallelDims) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Two copies of this rank's pipeline group (activations, gradients). Every rank creates every
    pipeline's copies, in the same order."""
    pp = parallel_dims.pp
    rank, mine = dist.get_rank(), None
    for ranks in parallel_dims.world_mesh.mesh.reshape(pp, -1).t().tolist():
        groups = (dist.new_group(ranks), dist.new_group(ranks))
        if rank in ranks:
            mine = groups
    return mine


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


def stage_layer_ids(num_layers: int, num_stages: int, stage: int, layers_per_stage: list[int] | None = None) -> range:
    """Consecutive layers of `stage`: `layers_per_stage` if given, else as even as possible with
    earlier stages taking the remainder."""
    if layers_per_stage is None:
        layers_per_stage = [
            num_layers // num_stages + (1 if i < num_layers % num_stages else 0) for i in range(num_stages)
        ]
    if len(layers_per_stage) != num_stages or sum(layers_per_stage) != num_layers:
        raise ValueError(f"layers_per_stage {layers_per_stage} must give {num_stages} stages {num_layers} layers")
    start = sum(layers_per_stage[:stage])
    return range(start, start + layers_per_stage[stage])


def prune_to_pipeline_stage(
    model: nn.Module, stage: int, num_stages: int, layers_per_stage: list[int] | None = None
) -> None:
    if not hasattr(model, "pipeline_stage_forward"):
        raise ValueError(f"{type(model).__name__} does not support pipeline parallelism")
    layer_ids = stage_layer_ids(len(model.model.layers), num_stages, stage, layers_per_stage)
    model.prune_to_pipeline_stage(layer_ids, first=stage == 0, last=stage == num_stages - 1)
    model.forward = model.pipeline_stage_forward


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
    # 1F1B keeps at most `pp` micro-batches per stage in flight; two spare sets cover receives
    # posted next to the previous micro-batch's backward. The multi-stage schedules keep at most
    # one per stage of the pipeline. GPipe keeps them all.
    pool = {"1F1B": pp + 2, "Async1F1B": pp + 2, "GPipe": None}.get(schedule, 2 * num_stages + 2)
    stage_cls = partial(PooledRecvPipelineStage, pool=pool) if pool is not None else PipelineStage
    stages = [
        stage_cls(
            model,
            stage_index=stage,
            num_stages=num_stages,
            device=torch.device("cuda", torch.cuda.current_device()),
            input_args=input_args,
            output_args=output_args,
            group=pp_mesh.get_group(),
        )
        for model, stage, (input_args, output_args) in zip(model_parts, stage_ids, shapes)
    ]
    # Gradients are scaled by the caller, like the gradient-accumulation path does.
    if schedule == "Async1F1B":
        fwd_group, bwd_group = directional_pipeline_groups(parallel_dims)
        return AsyncSchedule1F1B(
            stages[0],
            fwd_group=fwd_group,
            bwd_group=bwd_group,
            n_microbatches=num_micro_batches,
            loss_fn=loss_fn,
            scale_grads=False,
        )
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
