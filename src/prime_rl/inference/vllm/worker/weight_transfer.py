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
        # ``held_checksums`` kept current by ``apply_delta``; None until the next full recompute
        self.running: torch.Tensor | None = None
        # The fused kernel's device tables (``fused_tables``)
        self.fused = None
        # (name, write index) -> (row offsets, column offsets) for writes into a kernel layout that permutes rows or
        # columns (e.g. swizzled block scales): element (r, c) of the write is at its offset + rows[r] + cols[c], with
        # the write's stride for a missing table
        self.maps: dict[tuple[str, int], tuple[torch.Tensor | None, torch.Tensor | None]] = {}
        self.map_cache = None
        self.fp8: dict[str, str] = {}
        self.unpatchable = unpatchable
        # Checkpoint tensors this worker holds; unpatchable ones too, so a change to them is caught in apply_delta
        self.names = set(self.records)
        # Names that write the same elements (a tied weight's names, e.g. ``lm_head.weight`` of tied embeddings,
        # which HF checkpoints may omit) -> the other names
        by_writes = defaultdict(list)
        for name, writes in self.records.items():
            by_writes[tuple(writes)].append(name)
        self.tied = {
            name: [other for other in names if other != name]
            for names in by_writes.values()
            if len(names) > 1
            for name in names
        }

    def map_write(
        self, name: str, k: int, row_offsets: torch.Tensor | None = None, col_offsets: torch.Tensor | None = None
    ) -> None:
        """Place write ``k`` of ``name`` through offset tables (``maps``) instead of its strides."""
        _, rows, _, cols, *_ = self.records[name][k]
        assert row_offsets is None or row_offsets.shape == (rows,)
        assert col_offsets is None or col_offsets.shape == (cols,)
        self.maps[(name, k)] = (row_offsets, col_offsets)
        self.map_cache = self.checksum_plan = self.running = self.fused = None

    def _map_tables(self) -> tuple[torch.Tensor, dict[tuple[str, int], tuple[int, int]]]:
        """Every offset table in one int64 device buffer, and each mapped write's table starts (-1: none)."""
        if self.map_cache is None:
            device = self.params[0].device
            parts, starts, size = [torch.zeros(1, dtype=torch.int64)], {}, 1
            for key, tables in self.maps.items():
                first = []
                for table in tables:
                    if table is None:
                        first.append(-1)
                        continue
                    first.append(size)
                    parts.append(table.to(torch.int64).cpu())
                    size += table.numel()
                starts[key] = tuple(first)
            self.map_cache = (torch.cat(parts).to(device), starts)
        return self.map_cache

    def _targets(self, name: str, k: int) -> torch.Tensor:
        """Flat element offsets of write ``k`` of ``name`` in its param, ``[rows, cols]``."""
        _, rows, _, cols, _, offset, row_stride, col_stride = self.records[name][k]
        device = self.params[0].device
        row_table, col_table = self.maps.get((name, k), (None, None))
        row_part = row_table.to(device) if row_table is not None else torch.arange(rows, device=device) * row_stride
        col_part = col_table.to(device) if col_table is not None else torch.arange(cols, device=device) * col_stride
        return offset + row_part[:, None] + col_part[None, :]

    def checksums(self, full: bool = False) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """``held_checksums``, from the running sums ``apply_delta`` keeps unless ``full`` or nothing has
        recomputed them since the last load."""
        if full or self.running is None:
            return self.held_checksums()
        names, held, *_ = self._checksum_plan()
        return names, self.running.clone(), held

    def _checksum_plan(self):
        if self.checksum_plan is None:
            self.checksum_plan = self._plan_checksums()
        return self.checksum_plan

    def held_checksums(self) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """``bits_checksum`` of the live param elements each checkpoint tensor was written to, weighted by
        their checkpoint positions: the names, their sums, and how many elements of each this worker holds
        (fewer than the tensor's for TP slices, more workers than one for replicated KV heads). A full pass over
        the params; it also resets the running sums ``apply_delta`` updates."""
        names, held, row_ranges, other_writes, _, mapped = self._checksum_plan()
        totals = torch.zeros(len(names), dtype=torch.int64, device=held.device)
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
        for name_id, name, k in mapped:
            row0, rows, col0, cols, index, *_ = self.records[name][k]
            values = as_bits(self.params[index].data).view(-1)[self._targets(name, k)].long()
            num_cols = self.num_cols[name]
            row = torch.arange(rows, device=values.device)[:, None]
            col = torch.arange(cols, device=values.device)
            weights = (row0 + row) * num_cols + col0 + col + 1
            totals[name_id] += (values * weights).sum()
        self.running = totals.clone()
        return names, totals, held

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
        mapped = []
        for name_id, name in enumerate(names):
            num_cols = self.num_cols[name]
            for k, (row0, rows, col0, cols, index, offset, row_stride, col_stride) in enumerate(self.records[name]):
                width = self.params[index].shape[-1]
                if (name, k) in self.maps:
                    mapped.append((name_id, name, k))
                elif col_stride == 1 and row_stride == width and cols == width and offset % width == 0:
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
        held = torch.tensor(
            [sum(rows * cols for _, rows, _, cols, *_ in self.records[name]) for name in names], device=device
        )
        other_writes = {
            shape: tuple(torch.tensor(values, device=device) for values in lists)
            for shape, lists in other_writes.items()
        }
        plan_ids = {name: name_id for name_id, name in enumerate(names)}
        return names, held, row_ranges, other_writes, plan_ids, mapped

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
        self.checksum_plan = self.running = self.fused = None
        return self.fp8

    @torch.no_grad()
    def match_boot(self, boot_rows: Iterable[tuple[str, int, torch.Tensor]]) -> int:
        """Point every write at where the params hold its block, from the boot checkpoint's ``row_ranges`` rows
        (``(name, begin, rows)`` on the params' device) while the params still hold them. Kernel processing after
        loading can move blocks within a param (FlashInfer CUTLASS swaps the gate and up halves of fused expert
        weights); a write whose block is at another write's place in the same param takes that place. Raises if a
        block is nowhere. Before ``fp8_layout``; returns how many writes moved."""
        groups: dict[tuple[int, int, int], list[tuple[str, int]]] = defaultdict(list)
        for name, writes in self.records.items():
            for k, (_, rows, _, cols, index, *_) in enumerate(writes):
                groups[(index, rows, cols)].append((name, k))
        hashes: dict[tuple[int, int, int], dict[int, list[tuple[str, int]]]] = {}
        moves: dict[tuple[str, int], tuple[str, int]] = {}
        taken: set[tuple[str, int]] = set()
        for name, begin, rows in boot_rows:
            sources = {}
            for k, (row0, num_rows, col0, cols, index, *_) in enumerate(self.records[name]):
                dtype = self.params[index].dtype
                if dtype not in sources:
                    sources[dtype] = _boot_bits(rows, dtype)
                expected = [bits[row0 - begin : row0 - begin + num_rows, col0 : col0 + cols] for bits in sources[dtype]]
                if any(torch.equal(block, self._live(name, k)) for block in expected):
                    continue
                group = (index, num_rows, cols)
                if group not in hashes:
                    hashes[group] = defaultdict(list)
                    for slot in groups[group]:
                        hashes[group][_block_hash(self._live(*slot))].append(slot)
                found = [
                    slot
                    for block in expected
                    for slot in hashes[group].get(_block_hash(block), ())
                    if slot not in taken and torch.equal(block, self._live(*slot))
                ]
                if not found:
                    raise NotImplementedError(
                        f"Sparse weight deltas cannot patch {name}: the engine's {self.param_names[index]} does not "
                        "hold the boot checkpoint's block of it anywhere a write of the index map goes"
                    )
                moves[(name, k)] = found[0]
                taken.add(found[0])
        # The moved writes trade places among themselves; taking an unmoved write's place would put two writes there
        if taken != set(moves):
            raise NotImplementedError(
                f"Sparse weight deltas cannot patch {sorted({name for name, _ in moves})[:8]}: the engine's params "
                "hold their blocks in a layout that is not a permutation of the index map's writes"
            )
        places = {slot: (self.records[slot[0]][slot[1]][4:], self.maps.get(slot)) for slot in moves}
        for (name, k), slot in moves.items():
            place, tables = places[slot]
            self.records[name][k] = (*self.records[name][k][:4], *place)
            self.maps.pop((name, k), None)
            if tables is not None:
                self.maps[(name, k)] = tables
        self.map_cache = self.checksum_plan = self.running = self.fused = None
        return len(moves)

    def _live(self, name: str, k: int) -> torch.Tensor:
        """The raw bits the params hold at write ``k`` of ``name``, ``[rows, cols]``."""
        index = self.records[name][k][4]
        return as_bits(self.params[index].data).view(-1)[self._targets(name, k)]

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
            if any(self.params[write[4]].dtype == torch.float8_e4m3fn for write in writes):
                num_rows = self.numel[name] // self.num_cols[name]
                begin, end = begin // FP8_BLOCK[0] * FP8_BLOCK[0], min(-(-end // FP8_BLOCK[0]) * FP8_BLOCK[0], num_rows)
            ranges[name] = (begin, end)
        return ranges

    @torch.no_grad()
    def load_rows(self, name: str, rows: torch.Tensor, begin: int) -> None:
        """Set the params from rows ``begin:`` of checkpoint tensor ``name`` (the ``row_ranges`` rows, on the
        params' device): the counterpart of ``apply_delta`` for full checkpoints. fp8 tensors are quantized
        like the engine does, block scales included; tensors the engine casts on load are cast the same way."""
        self.running = None
        sources = [(name, rows, begin)]
        if name in self.fp8:
            quantized, scales = quantize_fp8_blocks(rows, self.fp8[name] == "ue8m0")
            sources = [(name, quantized, begin), (f"{name}_scale_inv", scales, begin // FP8_BLOCK[0])]
        for source_name, source, first in sources:
            for k, (row0, rows_, col0, cols, index, offset, *strides) in enumerate(self.records[source_name]):
                block = source[row0 - first : row0 - first + rows_, col0 : col0 + cols]
                if (source_name, k) in self.maps:
                    flat = as_bits(self.params[index].data).view(-1)
                    flat.index_put_((self._targets(source_name, k).view(-1),), as_bits(block).reshape(-1))
                elif source_name in self.unpatchable:
                    flat = self.params[index].data.view(-1)
                    flat.as_strided((rows_, cols), tuple(strides), offset).copy_(block)
                else:
                    self._view(index, rows_, cols, offset, strides).copy_(as_bits(block))

    def fused_tables(self, warm_up: bool = False) -> dict:
        """Device tables of ``sparse_apply``: tensor ids, every tensor's writes, columns and running-checksum id, and
        the params' addresses. Built once (and again after ``fp8_layout`` adds the block scales); ``warm_up`` also
        compiles the kernel, so neither happens inside a weight-update pause."""
        if self.fused is None:
            from prime_rl.inference.vllm.worker.sparse_apply import WRITE_FIELDS

            device = self.params[0].device
            names = sorted(self.records)
            max_writes = max((len(writes) for writes in self.records.values()), default=1)
            map_buffer, map_starts = self._map_tables()
            padding = [0] * (WRITE_FIELDS - 2) + [-1, -1]  # zero rows: never hit
            writes = torch.tensor(
                [
                    [[*write, *map_starts.get((name, k), (-1, -1))] for k, write in enumerate(self.records[name])]
                    + [padding] * (max_writes - len(self.records[name]))
                    for name in names
                ],
                dtype=torch.int64,
            ).view(len(names), max_writes, WRITE_FIELDS)
            plan_ids = self._checksum_plan()[4]
            self.fused = {
                "ids": {name: name_id for name_id, name in enumerate(names)},
                "writes": writes.to(device),
                "num_cols": torch.tensor([self.num_cols[name] for name in names], dtype=torch.int64, device=device),
                "plan_ids": torch.tensor([plan_ids.get(name, -1) for name in names], dtype=torch.int64, device=device),
                "param_ptrs": torch.tensor(
                    [param.data_ptr() for param in self.params], dtype=torch.int64, device=device
                ),
                "data_ptrs": [param.data_ptr() for param in self.params],
                "maps": map_buffer,
            }
            if warm_up:
                from prime_rl.inference.vllm.worker.sparse_apply import warm_up as compile_kernel

                compile_kernel(
                    self.fused["writes"], self.fused["num_cols"], self.fused["param_ptrs"], self.fused["maps"]
                )
        if [param.data_ptr() for param in self.params] != self.fused["data_ptrs"]:
            raise RuntimeError("Sparse weight deltas need the params where the index map found them: they moved")
        return self.fused

    @torch.no_grad()
    def apply_delta(self, entries: list[tuple[str, int]], indices: torch.Tensor, delta: torch.Tensor) -> None:
        """Add ``delta`` (raw-bit differences, wrapping in the bits dtype) to the params at the checkpoint
        ``indices`` of ``entries``."""
        changed = [name for name, _ in entries if name in self.unpatchable]
        if changed:
            raise NotImplementedError(f"Sparse weight deltas cannot patch {changed[:8]}: the engine casts them on load")
        device = indices.device
        if device.type == "cuda":
            self._apply_delta_fused(entries, indices, delta)
        else:
            self._apply_delta_torch(entries, indices, delta)

    def _apply_delta_torch(self, entries: list[tuple[str, int]], indices: torch.Tensor, delta: torch.Tensor) -> None:
        """``apply_delta`` in plain torch ops (CPU, and the reference for the fused kernel)."""
        device = indices.device
        max_records = max((len(self.records.get(name, ())) for name, _ in entries), default=0)
        map_buffer, map_starts = self._map_tables()
        # Per entry and write k: [num_cols, row0, rows, col0, cols, param, offset, row stride, col stride, row table,
        # column table].
        tables = [
            torch.tensor(
                [
                    [self.num_cols[name], *self.records[name][k], *map_starts.get((name, k), (-1, -1))]
                    if len(self.records.get(name, ())) > k
                    else [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1]
                    for name, _ in entries
                ],
                dtype=torch.int64,
                device=device,
            )
            for k in range(max_records)
        ]
        entry_ends = torch.tensor([count for _, count in entries], device=device).cumsum(0)
        # The running checksum of each entry's tensor (-1: not checksummed), updated with the changes alone
        plan_ids = self._checksum_plan()[4] if self.running is not None else None
        entry_ids = (
            torch.tensor([plan_ids.get(name, -1) for name, _ in entries], device=device)
            if plan_ids is not None
            else None
        )
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
                write, row, col = table[entry[hit]], row[hit], col[hit]
                last = map_buffer.numel() - 1
                row_part = torch.where(
                    write[:, 9] >= 0, map_buffer[(write[:, 9] + row).clamp(0, last)], row * write[:, 7]
                )
                col_part = torch.where(
                    write[:, 10] >= 0, map_buffer[(write[:, 10] + col).clamp(0, last)], col * write[:, 8]
                )
                target = write[:, 6] + row_part + col_part
                owner = write[:, 5]
                tracked = None if entry_ids is None else (entry_ids[entry][hit], index[hit] + 1)
                self._scatter_add(owner, target, source[hit], tracked)

    def _apply_delta_fused(self, entries: list[tuple[str, int]], indices: torch.Tensor, delta: torch.Tensor) -> None:
        from prime_rl.inference.vllm.worker.sparse_apply import sparse_apply

        tables = self.fused_tables()
        device = indices.device
        counts = torch.tensor([count for _, count in entries], dtype=torch.int64, device=device)
        entry_ends = counts.cumsum(0)
        entry_names = torch.tensor(
            [tables["ids"].get(name, -1) for name, _ in entries], dtype=torch.int64, device=device
        )
        track = self.running is not None
        contributions = sparse_apply(
            indices.long(),
            delta,
            entry_ends,
            entry_names,
            tables["writes"],
            tables["num_cols"],
            tables["param_ptrs"],
            tables["maps"],
            track,
        )
        if track and contributions.numel():
            # Per entry: a difference of the cumulative contributions at its ends (int64 sums wrap like the checksums)
            cumulative = torch.cat([contributions.new_zeros(1), contributions.cumsum(0)])
            sums = cumulative[entry_ends] - cumulative[entry_ends - counts]
            plan_ids = tables["plan_ids"][entry_names.clamp(min=0)]
            keep = (entry_names >= 0) & (plan_ids >= 0)
            self.running.index_add_(0, plan_ids[keep], sums[keep])

    def _scatter_add(
        self,
        owner: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        tracked: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Add ``source`` to the bits at ``target`` of params ``owner``. ``tracked`` (each change's checksum id
        and checkpoint position + 1) moves the running checksums by ``position * (new - old)``."""
        order = torch.argsort(owner, stable=True)
        owner, target, source = owner[order], target[order], source[order]
        if tracked is not None:
            ids, weights = (values[order] for values in tracked)
        counts = torch.bincount(owner, minlength=len(self.params)).tolist()
        end = 0
        for index, count in enumerate(counts):
            begin, end = end, end + count
            if not count:
                continue
            flat = as_bits(self.params[index].data).view(-1)
            positions = target[begin:end]
            old = flat[positions]
            new = old + source[begin:end]
            flat.index_put_((positions,), new)
            if tracked is not None:
                part_ids = ids[begin:end]
                keep = part_ids >= 0
                change = (new.long() - old.long()) * weights[begin:end]
                self.running.index_add_(0, part_ids[keep], change[keep])


def _boot_bits(rows: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor]:
    """The raw bits ``load_rows`` writes for checkpoint ``rows`` into a param of ``dtype``: fp8 in either scale
    format (``fp8_layout`` picks one later), others cast like the engine's loader."""
    if dtype == torch.float8_e4m3fn:
        return [quantize_fp8_blocks(rows, ue8m0)[0] for ue8m0 in (True, False)]
    return [as_bits(rows.to(dtype))]


def _block_hash(bits: torch.Tensor) -> int:
    """A position-weighted sum of a block's raw bits, to find a block among a param's writes."""
    flat = bits.reshape(-1).long()
    return int((flat * torch.arange(1, flat.numel() + 1, device=flat.device)).sum())


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
