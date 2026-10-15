"""Int8 wire format for the routed experts' FSDP collectives.

Each 128-value block of an expert weight row travels as int8 with one fp32 scale (absmax / 127, round to nearest
even). The all-gather dequantizes to bf16 on arrival, so compute, master weights and optimizer states are unchanged;
every rank sees the same dequantized weights. Gradients are quantized per destination rank, exchanged with an
all-to-all and summed in fp32 (NCCL cannot sum scaled int8). Other parameters in the same FSDP group travel unchanged.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import nn
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllocMixin
from triton.language.extra import libdevice

from prime_rl.trainer.model import get_language_model
from prime_rl.trainer.models.layers.moe import MoE

BLOCK = 128
BLOCKS_PER_PROGRAM = 8


@triton.jit
def _quantize_kernel(x_ptr, q_ptr, s_ptr, x_stride, q_stride, s_stride, n_blocks, BLOCKS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)
    cols = blocks[:, None] * 128 + tl.arange(0, 128)[None, :]
    mask = (blocks < n_blocks)[:, None]
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.math.div_rn(tl.max(tl.abs(x), axis=1), 127.0)
    q = tl.where(scale[:, None] > 0, libdevice.rint(tl.math.div_rn(x, scale[:, None])), 0.0)
    tl.store(q_ptr + row * q_stride + cols, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row * s_stride + blocks, scale, mask=blocks < n_blocks)


@triton.jit
def _dequantize_kernel(
    q_ptr, s_ptr, out_ptr, q_stride, s_stride, out_stride, n_rows, n_blocks, SUM: tl.constexpr, BLOCKS: tl.constexpr
):
    """SUM: out[0] = sum over rows of the dequantized rows (fp32 accumulation); else out[row] = row dequantized."""
    blocks = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    cols = blocks[:, None] * 128 + tl.arange(0, 128)[None, :]
    mask = (blocks < n_blocks)[:, None]
    if SUM:
        acc = tl.zeros((BLOCKS, 128), dtype=tl.float32)
        for row in range(n_rows):
            r = row.to(tl.int64)
            q = tl.load(q_ptr + r * q_stride + cols, mask=mask, other=0).to(tl.float32)
            scale = tl.load(s_ptr + r * s_stride + blocks, mask=blocks < n_blocks, other=0.0)
            acc += q * scale[:, None]
        tl.store(out_ptr + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        row = tl.program_id(1).to(tl.int64)
        q = tl.load(q_ptr + row * q_stride + cols, mask=mask, other=0).to(tl.float32)
        scale = tl.load(s_ptr + row * s_stride + blocks, mask=blocks < n_blocks, other=0.0)
        tl.store(out_ptr + row * out_stride + cols, (q * scale[:, None]).to(out_ptr.dtype.element_ty), mask=mask)


class Packing:
    """Byte layout of one rank's flat chunk: int8 values and scales of the quantized segments, then the rest raw.
    Chunks are rows of 2D tensors: one row per rank."""

    def __init__(self, segments: list[tuple[int, int, bool]], dtype: torch.dtype):
        self.dtype = dtype
        self.layout = []  # (chunk slice, quantized, value byte offset, scale byte offset)
        q_off = 0
        s_off = sum(numel for _, numel, q in segments if q)
        r_off = s_off + 4 * s_off // BLOCK
        for offset, numel, quantized in segments:
            position = slice(offset, offset + numel)
            if quantized:
                self.layout.append((position, True, q_off, s_off))
                q_off, s_off = q_off + numel, s_off + 4 * numel // BLOCK
            else:
                self.layout.append((position, False, r_off, None))
                r_off += numel * dtype.itemsize
        self.nbytes = -(-r_off // 16) * 16

    def _views(self, packed: torch.Tensor, position: slice, value_offset: int, scale_offset: int | None):
        numel = position.stop - position.start
        if scale_offset is None:
            return packed[:, value_offset : value_offset + numel * self.dtype.itemsize].view(self.dtype), None
        q = packed[:, value_offset : value_offset + numel].view(torch.int8)
        scale = packed[:, scale_offset : scale_offset + 4 * numel // BLOCK].view(torch.float32)
        return q, scale

    def pack(self, chunks: torch.Tensor) -> torch.Tensor:
        """(n, chunk numel) -> (n, nbytes) uint8."""
        packed = torch.empty(chunks.shape[0], self.nbytes, dtype=torch.uint8, device=chunks.device)
        for position, quantized, value_offset, scale_offset in self.layout:
            values, scale = self._views(packed, position, value_offset, scale_offset)
            x = chunks[:, position]
            if not quantized:
                values.copy_(x)
                continue
            n_blocks = x.shape[1] // BLOCK
            grid = (x.shape[0], triton.cdiv(n_blocks, BLOCKS_PER_PROGRAM))
            _quantize_kernel[grid](
                x, values, scale, x.stride(0), values.stride(0), scale.stride(0), n_blocks, BLOCKS_PER_PROGRAM
            )
        return packed

    def unpack(self, packed: torch.Tensor, out: torch.Tensor) -> None:
        """(n, nbytes) uint8 -> (n, chunk numel) ``out``."""
        for position, quantized, value_offset, scale_offset in self.layout:
            values, scale = self._views(packed, position, value_offset, scale_offset)
            target = out[:, position]
            if not quantized:
                target.copy_(values)
                continue
            n_blocks = target.shape[1] // BLOCK
            grid = (triton.cdiv(n_blocks, BLOCKS_PER_PROGRAM), target.shape[0])
            _dequantize_kernel[grid](
                values, scale, target, values.stride(0), scale.stride(0), target.stride(0), target.shape[0],
                n_blocks, False, BLOCKS_PER_PROGRAM,
            )  # fmt: skip

    def unpack_sum(self, packed: torch.Tensor, out: torch.Tensor) -> None:
        """(n, nbytes) uint8 -> fp32 sum over the n rows into the 1D ``out``."""
        for position, quantized, value_offset, scale_offset in self.layout:
            values, scale = self._views(packed, position, value_offset, scale_offset)
            target = out[position]
            if not quantized:
                target.copy_(values.float().sum(0))
                continue
            n_blocks = target.shape[0] // BLOCK
            _dequantize_kernel[(triton.cdiv(n_blocks, BLOCKS_PER_PROGRAM),)](
                values, scale, target, values.stride(0), scale.stride(0), 0, values.shape[0], n_blocks, True,
                BLOCKS_PER_PROGRAM,
            )  # fmt: skip


class Int8AllGather(DefaultAllocMixin, AllGather):
    def __init__(self, segments: list[tuple[int, int, bool]]):
        self.segments = segments

    def __call__(self, output_tensor, input_tensor, group, async_op=False):
        packing = Packing(self.segments, input_tensor.dtype)
        packed = packing.pack(input_tensor.view(1, -1))
        gathered = torch.empty(group.size(), packing.nbytes, dtype=torch.uint8, device=packed.device)
        dist.all_gather_into_tensor(gathered, packed, group=group)
        packing.unpack(gathered, output_tensor.view(group.size(), -1))


class Int8ReduceScatter(DefaultAllocMixin, ReduceScatter):
    """Sums only: the expert groups set ``force_sum_reduction_for_comms``, so FSDP divides afterwards."""

    def __init__(self, segments: list[tuple[int, int, bool]]):
        self.segments = segments

    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        assert op == dist.ReduceOp.SUM, op
        packing = Packing(self.segments, input_tensor.dtype)
        packed = packing.pack(input_tensor.view(group.size(), -1))
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed, group=group)
        packing.unpack_sum(received, output_tensor)


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
