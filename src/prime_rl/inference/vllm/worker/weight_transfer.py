from collections import defaultdict
from math import prod
from typing import Iterable

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload
from vllm.model_executor.model_loader.reload.layerwise import (
    LAYERWISE_INFO,
    _get_original_loader,
    _get_weight_loader,
)
from vllm.model_executor.model_loader.reload.meta import restore_layer_on_meta
from vllm.model_executor.model_loader.reload.utils import get_layer_params_buffers, get_layer_tensors

from prime_rl.utils.sparse_weights import FP8_BLOCK, as_bits, quantize_fp8_blocks

logger = init_logger("vllm.inference.vllm.worker_weight_transfer")


def load_weights_checkpoint_layerwise(
    model: Module,
    state_iter: Iterable[tuple[str, torch.Tensor]],
    model_config,
    vllm_config,
) -> None:
    logger.info("Reloading checkpoint-format weights with vLLM layerwise processing")
    device = next(model.parameters()).device
    with torch.device(device), set_current_vllm_config(vllm_config):
        initialize_layerwise_reload(model)
        model.load_weights(state_iter)  # type: ignore
        finalize_layerwise_reload(model, model_config)


class CheckpointIndexMap:
    """Maps flat checkpoint (HF) indices to flat indices of the live vLLM params.

    Built once by running the model's own loader on placeholder checkpoint
    tensors with ``Tensor.copy_`` replaced by a recorder, so it covers the
    loader's name mapping, packing (qkv, gate_up, experts) and TP/EP slicing.
    Layers quantized online (e.g. ``fp8_per_block``) are recorded in their
    checkpoint format; their live params must keep that layout (one byte per
    checkpoint element). Supported: every write is one ``copy_`` of a row/column
    block of a 1-D or 2-D checkpoint tensor. Tensors the loader casts (e.g. an
    fp32 norm from a bf16 checkpoint) are loaded by value from full checkpoints
    and anchors; a delta that changes them raises.
    """

    def __init__(self, model: Module, tensors: dict[str, tuple[list[int], str]]):
        device = next(model.parameters()).device
        source_names: dict[int, str] = {}
        # name -> [(row0, rows, col0, cols, param name, offset, row stride, col stride)]
        records: dict[str, list[tuple]] = defaultdict(list)
        shapes: dict[str, torch.Size] = {}
        unpatchable: set[str] = set()

        def record_copy_(destination: torch.Tensor, source: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
            name = source_names.get(source.untyped_storage()._cdata)
            param_name = param_names.get(destination.untyped_storage()._cdata)
            shape = tensors[name][0] if name is not None else None
            num_cols = shape[-1] if shape else 1
            source_2d = source.view(1, -1) if source.dim() == 1 else source
            destination_2d = destination.view(1, -1) if destination.dim() == 1 else destination
            if name is not None and source.dtype != destination.dtype:
                # Cast on load (e.g. an fp32 norm from a bf16 checkpoint): loaded by value, no deltas
                unpatchable.add(name)
            if (
                name is None
                or param_name is None
                or len(shape) > 2
                or source.shape != destination.shape
                or source.dim() > 2
                or source_2d.stride(-1) != 1
                or (source_2d.shape[0] > 1 and source_2d.stride(0) != num_cols)
            ):
                raise NotImplementedError("Sparse weight deltas need plain block copies from checkpoint tensors")
            row0, col0 = divmod(source.storage_offset(), num_cols)
            records[name].append(
                (row0, source_2d.shape[0], col0, source_2d.shape[1], param_name, destination.storage_offset())
                + tuple(destination_2d.stride())
            )
            return destination

        def placeholders():
            for name, (shape, dtype) in tensors.items():
                if name.endswith("_scale_inv"):
                    continue  # fp8 block scales of a delta: recorded in fp8_layout
                # fp8 tensors of a delta are quantized from bf16 checkpoint tensors
                checkpoint_dtype = torch.bfloat16 if dtype == "float8_e4m3fn" else getattr(torch, dtype)
                tensor = torch.empty(shape, dtype=checkpoint_dtype, device=device)
                source_names[tensor.untyped_storage()._cdata] = name
                yield name, tensor

        swapped = _restore_checkpoint_format(model)
        original_copy = torch.Tensor.copy_
        try:
            param_names = {}
            for name, param in model.named_parameters():
                if param.storage_offset() == 0 and param.is_contiguous():
                    param_names.setdefault(param.untyped_storage()._cdata, name)
                    shapes[name] = param.shape
            torch.Tensor.copy_ = record_copy_
            model.load_weights(placeholders())  # type: ignore
        finally:
            torch.Tensor.copy_ = original_copy
            _restore_kernel_format(swapped)

        live = dict(model.named_parameters())
        used = list(dict.fromkeys(write[4] for writes in records.values() for write in writes))
        for name in used:
            if live[name].shape != shapes[name] or not live[name].is_contiguous() or live[name].storage_offset():
                raise NotImplementedError(f"Sparse weight deltas cannot patch {name}: its layout changes after loading")
        self.params: list[torch.Tensor] = [live[name] for name in used]
        self.param_names = used
        param_index = {name: index for index, name in enumerate(used)}
        self.records = {
            name: [(*write[:4], param_index[write[4]], *write[5:]) for write in writes]
            for name, writes in records.items()
        }
        self.num_cols = {name: (shape[-1] if shape else 1) for name, (shape, _) in tensors.items()}
        self.numel = {name: prod(shape) for name, (shape, _) in tensors.items()}
        self.checksum_plan = None
        self.fp8: dict[str, str] = {}
        self.unpatchable = unpatchable
        # Checkpoint tensors this worker holds; unpatchable ones too, so a change to them is caught in apply_delta
        self.names = set(self.records)

    def held_checksums(self) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """``bits_checksum`` of the live param elements each checkpoint tensor was written to, weighted by
        their checkpoint positions: the names, their sums, and a mask of the ones this worker holds only
        part of (TP slices)."""
        if self.checksum_plan is None:
            self.checksum_plan = self._plan_checksums()
        names, partial, row_ranges, other_writes = self.checksum_plan
        totals = torch.zeros(len(names), dtype=torch.int64, device=partial.device)
        for index, (starts, ends, ids, num_cols, firsts) in row_ranges.items():
            bits = as_bits(self.params[index].data)
            bits = bits.reshape(-1, bits.shape[-1])
            row_sums = bits.sum(dim=1, dtype=torch.int64)
            # Σ_c c·bits[row, c], in blocks of rows that bound the int64 temporary
            columns = torch.arange(bits.shape[1], device=bits.device)
            moments = torch.cat([(block * columns).sum(1) for block in bits.split(max(1, (1 << 24) // bits.shape[1]))])
            rows = torch.arange(bits.shape[0], device=bits.device)
            plain, by_row, by_col = (
                cumulative[ends] - cumulative[starts]
                for cumulative in (
                    torch.cat([values.new_zeros(1), values.cumsum(0)])
                    for values in (row_sums, rows * row_sums, moments)
                )
            )
            totals.index_add_(0, ids, num_cols * by_row + firsts * plain + by_col)
        # Writes of one shape into one param at once, e.g. the block scales of every expert
        for (index, rows, cols, row_stride, col_stride), (ids, num_cols, firsts, offsets) in other_writes.items():
            flat = as_bits(self.params[index].data).view(-1)
            row = torch.arange(rows, device=flat.device)[:, None]
            col = torch.arange(cols, device=flat.device)
            for batch in range(0, len(ids), max(1, (1 << 24) // (rows * cols))):
                part = slice(batch, batch + max(1, (1 << 24) // (rows * cols)))
                values = flat[offsets[part, None, None] + row * row_stride + col * col_stride].long()
                weights = firsts[part, None, None] + num_cols[part, None, None] * row + col
                totals.index_add_(0, ids[part], (values * weights).sum((1, 2)))
        return names, totals, partial

    def _plan_checksums(self):
        """Writes that cover whole rows of a param become row ranges, summed with cumsums over the rows
        of each param. Element ``(row, col)`` of a write is weighted by its checkpoint flat index + 1:
        ``row * num_cols + first + col`` for a row range, with ``first`` per write."""
        names = [
            name
            for name, writes in self.records.items()
            if name not in self.unpatchable and sum(rows * cols for _, rows, _, cols, *_ in writes) <= self.numel[name]
        ]
        device = self.params[0].device
        ranges: dict[int, list[list[int]]] = defaultdict(lambda: [[], [], [], [], []])
        other_writes: dict[tuple, list[list[int]]] = defaultdict(lambda: [[], [], [], []])
        for name_id, name in enumerate(names):
            num_cols = self.num_cols[name]
            for row0, rows, col0, cols, index, offset, row_stride, col_stride in self.records[name]:
                width = self.params[index].shape[-1]
                if col_stride == 1 and row_stride == width and cols == width and offset % width == 0:
                    start = offset // width
                    for values, value in zip(
                        ranges[index], (start, start + rows, name_id, num_cols, (row0 - start) * num_cols + col0 + 1)
                    ):
                        values.append(value)
                else:
                    batch = other_writes[(index, rows, cols, row_stride, col_stride)]
                    for values, value in zip(batch, (name_id, num_cols, row0 * num_cols + col0 + 1, offset)):
                        values.append(value)
        row_ranges = {
            index: tuple(torch.tensor(values, device=device) for values in lists) for index, lists in ranges.items()
        }
        held = [sum(rows * cols for _, rows, _, cols, *_ in self.records[name]) for name in names]
        partial = torch.tensor([h < self.numel[name] for h, name in zip(held, names)], device=device)
        other_writes = {
            shape: tuple(torch.tensor(values, device=device) for values in lists)
            for shape, lists in other_writes.items()
        }
        return names, partial, row_ranges, other_writes

    def fp8_layout(self, model: Module, read_tensor) -> dict[str, str]:
        """The vLLM per-block quantization (``"fp32"`` or ``"ue8m0"`` scales) that reproduces each
        fp8 param from the checkpoint, per checkpoint tensor stored in one. Also records where each such
        tensor's block scales (``"{name}_scale_inv"``, one fp32 per 128x128 block) live, so deltas and loads
        patch them in place. Checks one checkpoint tensor per param (weights and scales); ``read_tensor(name)``
        returns it."""
        modes: dict[int, str] = {}
        live = dict(model.named_parameters())
        scale_params: dict[str, int] = {}
        for name, writes in list(self.records.items()):
            for row0, rows, col0, cols, index, offset, *strides in writes:
                param = self.params[index]
                if param.dtype != torch.float8_e4m3fn:
                    continue
                scale = live.get(f"{self.param_names[index]}_scale_inv")
                num_rows, num_cols = param.shape[-2:]
                row, col = offset // num_cols % num_rows, offset % num_cols
                if (
                    scale is None
                    or scale.dtype != torch.float32
                    or scale.shape[:-2] != param.shape[:-2]
                    or strides != [num_cols, 1]
                    or any(x % FP8_BLOCK[0] for x in (row0, col0, row, col))
                ):
                    raise NotImplementedError(f"Sparse weight deltas cannot patch the fp8 block scales of {name}")
                if scale.untyped_storage()._cdata not in scale_params:
                    scale_params[scale.untyped_storage()._cdata] = len(self.params)
                    # The whole storage, flat: kernels may keep scales in a strided (e.g. MN-major) layout
                    self.params.append(scale.data.as_strided((scale.untyped_storage().nbytes() // 4,), (1,), 0))
                matrix, scale_offset = offset // (num_rows * num_cols), scale.storage_offset()
                for dim in reversed(range(param.dim() - 2)):
                    matrix, position = divmod(matrix, param.shape[dim])
                    scale_offset += position * scale.stride(dim)
                scale_offset += row // FP8_BLOCK[0] * scale.stride(-2) + col // FP8_BLOCK[1] * scale.stride(-1)
                block = (row0 // FP8_BLOCK[0], -(-rows // FP8_BLOCK[0]), col0 // FP8_BLOCK[1], -(-cols // FP8_BLOCK[1]))
                scale_name = f"{name}_scale_inv"
                self.records.setdefault(scale_name, []).append(
                    (*block, scale_params[scale.untyped_storage()._cdata], scale_offset, *scale.stride()[-2:])
                )
                self.num_cols[scale_name] = -(-self.num_cols[name] // FP8_BLOCK[1])
                self.numel[scale_name] = (
                    -(-self.numel[name] // self.num_cols[name] // FP8_BLOCK[0]) * self.num_cols[scale_name]
                )
                if index not in modes:
                    weight = read_tensor(name).to(param.device)
                    weight = weight.view(-1, self.num_cols[name])
                    for mode in ("ue8m0", "fp32"):
                        quantized, scales = quantize_fp8_blocks(weight, mode == "ue8m0")
                        if torch.equal(
                            quantized[row0 : row0 + rows, col0 : col0 + cols],
                            self._view(index, rows, cols, offset, strides),
                        ):
                            modes[index] = mode
                            break
                    else:
                        raise NotImplementedError(
                            f"Sparse weight deltas cannot patch {name}: the engine's fp8 weights are not vLLM's "
                            "per-block quantization of the checkpoint (e.g. a weight-shuffling kernel)"
                        )
                    _, block_rows, _, block_cols, scale_index, scale_offset, *scale_strides = self.records[scale_name][
                        -1
                    ]
                    expected = scales[block[0] : block[0] + block_rows, block[2] : block[2] + block_cols]
                    if not torch.equal(
                        as_bits(expected), self._view(scale_index, block_rows, block_cols, scale_offset, scale_strides)
                    ):
                        raise NotImplementedError(f"Sparse weight deltas cannot patch the fp8 block scales of {name}")
                if self.fp8.setdefault(name, modes[index]) != modes[index]:
                    raise NotImplementedError(f"Sparse weight deltas cannot patch {name}: mixed fp8 layouts")
        self.names |= set(self.records)
        self.checksum_plan = None
        return self.fp8

    def _view(self, index: int, rows: int, cols: int, offset: int, strides) -> torch.Tensor:
        """The raw bits of one write of a checkpoint block into param ``index``."""
        return as_bits(self.params[index].data).view(-1).as_strided((rows, cols), tuple(strides), offset)

    def row_ranges(self) -> dict[str, tuple[int, int]]:
        """Checkpoint tensor -> the rows ``[begin, end)`` of its ``[rows, num_cols]`` view this worker holds
        (fp8 tensors: whole 128-row blocks), for ``load_rows``."""
        ranges = {}
        for name, writes in self.records.items():
            if name.endswith("_scale_inv"):
                continue  # scales are set with their weights
            begin, end = min(write[0] for write in writes), max(write[0] + write[1] for write in writes)
            if name in self.fp8:
                num_rows = self.numel[name] // self.num_cols[name]
                begin, end = begin // FP8_BLOCK[0] * FP8_BLOCK[0], min(-(-end // FP8_BLOCK[0]) * FP8_BLOCK[0], num_rows)
            ranges[name] = (begin, end)
        return ranges

    @torch.no_grad()
    def load_rows(self, name: str, rows: torch.Tensor, begin: int) -> None:
        """Set the params from rows ``begin:`` of checkpoint tensor ``name`` (the ``row_ranges`` rows, on the
        params' device): the counterpart of ``apply_delta`` for full checkpoints. fp8 tensors are quantized
        like the engine does, block scales included; tensors the engine casts on load are cast the same way."""
        sources = [(name, rows, begin)]
        if name in self.fp8:
            quantized, scales = quantize_fp8_blocks(rows, self.fp8[name] == "ue8m0")
            sources = [(name, quantized, begin), (f"{name}_scale_inv", scales, begin // FP8_BLOCK[0])]
        for source_name, source, first in sources:
            for row0, rows_, col0, cols, index, offset, *strides in self.records[source_name]:
                block = source[row0 - first : row0 - first + rows_, col0 : col0 + cols]
                if source_name in self.unpatchable:
                    flat = self.params[index].data.view(-1)
                    flat.as_strided((rows_, cols), tuple(strides), offset).copy_(block)
                else:
                    self._view(index, rows_, cols, offset, strides).copy_(as_bits(block))

    @torch.no_grad()
    def apply_delta(self, entries: list[tuple[str, int]], indices: torch.Tensor, delta: torch.Tensor) -> None:
        """Add ``delta`` (raw-bit differences, wrapping in the bits dtype) to the params at the checkpoint
        ``indices`` of ``entries``."""
        changed = [name for name, _ in entries if name in self.unpatchable]
        if changed:
            raise NotImplementedError(f"Sparse weight deltas cannot patch {changed[:8]}: the engine casts them on load")
        device = indices.device
        max_records = max((len(self.records.get(name, ())) for name, _ in entries), default=0)
        # Per entry and write k: [num_cols, row0, rows, col0, cols, param, offset, row stride, col stride].
        tables = [
            torch.tensor(
                [
                    [self.num_cols[name], *self.records[name][k]]
                    if len(self.records.get(name, ())) > k
                    else [1, 0, 0, 0, 0, 0, 0, 0, 0]
                    for name, _ in entries
                ],
                dtype=torch.int64,
                device=device,
            )
            for k in range(max_records)
        ]
        entry_ends = torch.tensor([count for _, count in entries], device=device).cumsum(0)
        chunk = 1 << 25
        for start in range(0, indices.numel(), chunk):
            index = indices[start : start + chunk]
            source = delta[start : start + chunk]
            entry = torch.searchsorted(
                entry_ends, torch.arange(start, start + index.numel(), device=device), right=True
            )
            for table in tables:
                num_cols = table[entry, 0]
                row = index // num_cols - table[entry, 1]
                col = index % num_cols - table[entry, 3]
                hit = (row >= 0) & (row < table[entry, 2]) & (col >= 0) & (col < table[entry, 4])
                target = (table[entry, 6] + row * table[entry, 7] + col * table[entry, 8])[hit]
                owner = table[entry, 5][hit]
                self._scatter_add(owner, target, source[hit])

    def _scatter_add(self, owner: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> None:
        order = torch.argsort(owner, stable=True)
        owner, target, source = owner[order], target[order], source[order]
        counts = torch.bincount(owner, minlength=len(self.params)).tolist()
        end = 0
        for index, count in enumerate(counts):
            begin, end = end, end + count
            if not count:
                continue
            flat = as_bits(self.params[index].data).view(-1)
            positions = target[begin:end]
            flat.index_put_((positions,), flat[positions] + source[begin:end])


def _restore_checkpoint_format(model: Module) -> list[tuple]:
    """Put the checkpoint-format params (on meta, with their unwrapped loaders) back into
    every layer whose live params were transformed after loading (online quantization)."""
    swapped = []
    for layer in model.modules():
        info = LAYERWISE_INFO.get(layer)
        if info is None:
            continue
        restore_params, _ = info.restore_metadata
        live = layer._parameters
        if all(
            live.get(name) is not None and (live[name].dtype, live[name].shape) == (param.dtype, param.shape)
            for name, param in restore_params.items()
        ):
            continue
        kernel_tensors = get_layer_params_buffers(layer)
        non_persistent = set(layer._non_persistent_buffers_set)
        restore_layer_on_meta(layer, info)
        loaders = {}
        for name, param in get_layer_params_buffers(layer)[0].items():
            loaders[name] = _get_weight_loader(param)
            param.weight_loader = _get_original_loader(param)
        swapped.append((layer, kernel_tensors, non_persistent, loaders))
    return swapped


def _restore_kernel_format(swapped: list[tuple]) -> None:
    for layer, (params, buffers), non_persistent, loaders in swapped:
        for name, loader in loaders.items():
            getattr(layer, name).weight_loader = loader
        for name in get_layer_tensors(layer):
            delattr(layer, name)
        for name, param in params.items():
            layer.register_parameter(name, param)
        for name, buffer in buffers.items():
            layer.register_buffer(name, buffer, persistent=name not in non_persistent)


@torch.no_grad()
def update_mla_absorbed_weights(model: Module) -> None:
    """Recompute MLA absorbed KV weights after in-place kv_b_proj updates."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import get_and_maybe_dequant_weights

    for name, module in model.named_modules():
        has_absorbed_weights = hasattr(module, "W_UV") or hasattr(module, "W_UK_T")
        if not has_absorbed_weights or not hasattr(module, "kv_b_proj"):
            continue

        if hasattr(module, "W_UV"):
            out_dtype = module.W_UV.dtype
        else:
            out_dtype = torch.bfloat16

        kv_b_proj_weight = get_and_maybe_dequant_weights(module.kv_b_proj, out_dtype=out_dtype).T
        kv_b_proj_weight = kv_b_proj_weight.view(
            module.kv_lora_rank,
            module.num_heads,
            module.qk_nope_head_dim + module.v_head_dim,
        )
        w_uk, w_uv = kv_b_proj_weight.split([module.qk_nope_head_dim, module.v_head_dim], dim=-1)

        if hasattr(module, "W_UV"):
            module.W_UV.copy_(w_uv.transpose(0, 1))
        if hasattr(module, "W_UK_T"):
            module.W_UK_T.copy_(w_uk.permute(1, 2, 0))

        logger.debug(f"Updated MLA absorbed weights for module {name}")
