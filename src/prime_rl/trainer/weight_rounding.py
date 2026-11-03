import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from types import EllipsisType
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

# Rows of a parameter rounded at once, about this many elements: bounds the temporary noise and comparison tensors
CHUNK_NUMEL = 1 << 24
LOW_BITS = 0xFFFF


Mode = Literal["nearest", "stochastic", "dithered"]


def round_to_bf16(master: torch.Tensor, mode: Mode, generator: torch.Generator | None = None) -> torch.Tensor:
    """The bf16 rounding of fp32 ``master``, as fp32. ``stochastic`` and ``dithered`` round each value up with
    probability proportional to its distance from the bf16 value below it (in magnitude), so the rounding is
    unbiased; they differ only in how ``WeightRounding`` seeds the noise."""
    if mode == "nearest":
        return master.to(torch.bfloat16).float()
    noise = torch.randint(0, LOW_BITS + 1, master.shape, dtype=torch.int32, device=master.device, generator=generator)
    # Adding up to one bf16 ulp to the low 16 bits carries into bit 16 with the right probability; the sign bit is
    # separate, so this rounds the magnitude and stays unbiased for negative values
    return ((master.view(torch.int32) + noise) & ~LOW_BITS).view(torch.float32)


class WeightRounding:
    """Holds the fp32 master weights apart from the parameters, which hold the masters' bf16 rounding.

    FSDP casts the parameters to bf16 for the forward and the weight broadcast casts them to bf16 for inference, so
    both see exactly the rounded weights, while the optimizer steps the masters: a straight-through estimator.
    ``nearest`` rounding reproduces the default numerics and only adds the metrics. The other modes are unbiased, so
    updates below half a bf16 ulp still reach the trained and served weights in expectation instead of accumulating
    unseen in the masters until they cross a rounding boundary. ``stochastic`` draws new noise every step, so a
    weight's rounding is unbiased over steps but flips back and forth even when its master stays put. ``dithered``
    keeps each weight's noise across steps: a weight's rounding changes only when its master crosses its own random
    threshold, so the fraction of weights that change follows how far the masters moved.

    The noise depends on the seed, the step (``stochastic`` only), the parameter and the global position of the local
    rows, so replicas of a shard round alike and a resume reproduces the rounding of the step it resumes after.
    """

    def __init__(self, model: nn.Module, mode: Mode, seed: int, step: int):
        """Takes the parameters as the masters and rounds them as the rounding of step ``step`` did."""
        self.mode = mode
        self.seed = seed
        self.params: list[tuple[int, torch.Tensor, tuple[int, ...]]] = []
        for name, param in model.named_parameters():
            if not param.requires_grad or param.dtype != torch.float32:
                continue
            offset = (0,) * param.ndim
            if isinstance(param, DTensor):
                offset = tuple(
                    compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)[1]
                )
            self.params.append((zlib.crc32(name.encode()), _local(param), offset))
        self.masters = [local.clone() for _, local, _ in self.params]
        self.generator = torch.Generator(device="cuda")
        self.step = step
        with torch.no_grad():
            for (key, local, offset), master in zip(self.params, self.masters):
                for rows, first_row in _row_chunks(master):
                    local[rows].copy_(self._round(master[rows], key, offset, first_row, step))

    def _round(
        self, master: torch.Tensor, key: int, offset: tuple[int, ...], first_row: int, step: int
    ) -> torch.Tensor:
        if self.mode != "nearest":
            step_key = step if self.mode == "stochastic" else 0
            self.generator.manual_seed(hash((self.seed, step_key, key, offset, first_row)) & ((1 << 63) - 1))
        return round_to_bf16(master, self.mode, self.generator)

    @torch.no_grad()
    def restore_masters(self) -> None:
        """Puts the masters into the parameters, for the optimizer step."""
        for (_, local, _), master in zip(self.params, self.masters):
            local.copy_(master)

    @torch.no_grad()
    def round(self, step: int) -> dict[str, float]:
        """After the optimizer step of ``step``: takes the parameters as the masters, puts their rounding into the
        parameters and returns how the rounded weights follow the masters."""
        device = torch.device("cuda", torch.cuda.current_device())
        # Changed rounded values, values, squared master-to-rounded gap, squared master update
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        for (key, local, offset), master in zip(self.params, self.masters):
            for rows, first_row in _row_chunks(master):
                old_master, new_master = master[rows], local[rows]
                old = self._round(old_master, key, offset, first_row, self.step)
                new = self._round(new_master, key, offset, first_row, step)
                totals[0] += (new != old).sum()
                totals[1] += new.numel()
                totals[2] += (new_master - new).double().square().sum()
                totals[3] += (new_master - old_master).double().square().sum()
                old_master.copy_(new_master)
                new_master.copy_(new)
        self.step = step
        dist.all_reduce(totals)
        changed, numel, gap, update = totals.tolist()
        return {
            "weight_rounding/changed_frac": changed / numel,
            "weight_rounding/lag": (gap / update) ** 0.5 if update > 0 else 0.0,
        }

    @contextmanager
    def masters_in_params(self) -> Iterator[None]:
        """Puts the masters into the parameters for the duration, e.g. of a checkpoint save."""
        self.restore_masters()
        try:
            yield
        finally:
            with torch.no_grad():
                for (key, local, offset), master in zip(self.params, self.masters):
                    for rows, first_row in _row_chunks(master):
                        local[rows].copy_(self._round(master[rows], key, offset, first_row, self.step))


def _local(param: torch.Tensor) -> torch.Tensor:
    return param.data.to_local() if isinstance(param, DTensor) else param.data


def _row_chunks(tensor: torch.Tensor) -> Iterator[tuple[slice | EllipsisType, int]]:
    if tensor.ndim == 0:
        yield ..., 0
        return
    rows = max(1, CHUNK_NUMEL // max(1, tensor[0].numel()))
    for first in range(0, tensor.shape[0], rows):
        yield slice(first, first + rows), first
