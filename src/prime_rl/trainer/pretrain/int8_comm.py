"""Int8 wire format for the routed experts' FSDP collectives.

Each 128-value block of an expert weight row travels as int8 with one fp32 scale (absmax / 127). The all-gather
dequantizes to bf16 on arrival, so compute, master weights and optimizer states are unchanged; every rank sees the
same dequantized weights. Gradients are quantized per destination rank, exchanged with an all-to-all and summed in
fp32 (NCCL cannot sum scaled int8). Other parameters in the same FSDP group travel unchanged.
"""

from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllocMixin

from prime_rl.trainer.model import get_language_model
from prime_rl.trainer.models.layers.moe import MoE

BLOCK = 128


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of ``x`` (last dim a multiple of BLOCK) -> int8 values and one fp32 scale per block."""
    blocks = x.float().unflatten(-1, (-1, BLOCK))
    scale = blocks.abs().amax(-1, keepdim=True) / 127
    q = torch.where(scale > 0, blocks / scale, 0).round().to(torch.int8)
    return q.flatten(-2), scale.squeeze(-1)


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.unflatten(-1, (-1, BLOCK)).float() * scale.unsqueeze(-1)).flatten(-2)


class Packing:
    """Byte layout of one rank's flat chunk: int8 values and scales of the quantized segments, then the rest raw."""

    def __init__(self, segments: list[tuple[int, int, bool]], dtype: torch.dtype):
        self.segments = segments  # (offset, numel, quantized) in the flat chunk
        self.dtype = dtype
        quantized = [numel for _, numel, q in segments if q]
        raw = [numel for _, numel, q in segments if not q]
        self.q_bytes = sum(quantized)
        self.scale_bytes = 4 * self.q_bytes // BLOCK
        self.nbytes = -(-(self.q_bytes + self.scale_bytes + dtype.itemsize * sum(raw)) // 16) * 16

    def pack(self, chunks: torch.Tensor) -> torch.Tensor:
        """(n, chunk numel) -> (n, nbytes) uint8."""
        out = torch.zeros(chunks.shape[0], self.nbytes, dtype=torch.uint8, device=chunks.device)
        q_off, s_off, r_off = 0, self.q_bytes, self.q_bytes + self.scale_bytes
        for offset, numel, quantized in self.segments:
            x = chunks[:, offset : offset + numel]
            if quantized:
                q, scale = quantize(x)
                out[:, q_off : q_off + numel] = q.view(torch.uint8)
                out[:, s_off : s_off + scale.shape[1] * 4] = scale.view(torch.uint8)
                q_off, s_off = q_off + numel, s_off + scale.shape[1] * 4
            else:
                nbytes = numel * self.dtype.itemsize
                out[:, r_off : r_off + nbytes] = x.contiguous().view(torch.uint8)
                r_off += nbytes
        return out

    def unpack(self, packed: torch.Tensor) -> Iterator[tuple[slice, torch.Tensor]]:
        """(n, nbytes) uint8 -> each segment's position in the chunk and its (n, numel) values."""
        q_off, s_off, r_off = 0, self.q_bytes, self.q_bytes + self.scale_bytes
        for offset, numel, quantized in self.segments:
            if quantized:
                q = packed[:, q_off : q_off + numel].view(torch.int8)
                scale = packed[:, s_off : s_off + numel // BLOCK * 4].view(torch.float32)
                yield slice(offset, offset + numel), dequantize(q, scale)
                q_off, s_off = q_off + numel, s_off + numel // BLOCK * 4
            else:
                nbytes = numel * self.dtype.itemsize
                yield slice(offset, offset + numel), packed[:, r_off : r_off + nbytes].view(self.dtype)
                r_off += nbytes


class Int8AllGather(DefaultAllocMixin, AllGather):
    def __init__(self, segments: list[tuple[int, int, bool]]):
        self.segments = segments

    def __call__(self, output_tensor, input_tensor, group, async_op=False):
        packing = Packing(self.segments, input_tensor.dtype)
        packed = packing.pack(input_tensor.view(1, -1))
        gathered = torch.empty(group.size(), packing.nbytes, dtype=torch.uint8, device=packed.device)
        dist.all_gather_into_tensor(gathered, packed, group=group)
        output = output_tensor.view(group.size(), -1)
        for position, values in packing.unpack(gathered):
            output[:, position] = values


class Int8ReduceScatter(DefaultAllocMixin, ReduceScatter):
    """Sums only: the expert groups set ``force_sum_reduction_for_comms``, so FSDP divides afterwards."""

    def __init__(self, segments: list[tuple[int, int, bool]]):
        self.segments = segments

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        assert op == dist.ReduceOp.SUM, op
        packing = Packing(self.segments, input_tensor.dtype)
        chunks = input_tensor.view(group.size(), -1)
        packed = packing.pack(chunks)
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed, group=group)
        for position, values in packing.unpack(received):
            output_tensor[position] = values.float().sum(0)


def apply_int8_expert_comm(model: nn.Module, grads: bool) -> None:
    """Send the routed experts' weights (and with ``grads`` their gradients) as blockwise int8 in every FSDP group
    that holds them. Call after FSDP is applied."""
    for layer in get_language_model(model).layers:
        if not isinstance(getattr(layer, "mlp", None), MoE):
            continue
        expert_params = {id(param) for param in layer.mlp.experts.parameters()}
        for group in layer._get_fsdp_state()._fsdp_param_groups:
            segments, offset = [], 0
            for fsdp_param in group.fsdp_params:
                numel = fsdp_param.padded_sharded_param_size.numel()
                quantized = id(fsdp_param.sharded_param) in expert_params
                assert not quantized or numel % BLOCK == 0, numel
                segments.append((offset, numel, quantized))
                offset += numel
            if not any(quantized for _, _, quantized in segments):
                continue
            group._all_gather_comm = Int8AllGather(segments)
            if grads:
                group._reduce_scatter_comm = Int8ReduceScatter(segments)
                group.force_sum_reduction_for_comms = True
