from collections import defaultdict
from typing import Iterable

import torch
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

from prime_rl.utils.sparse_weights import as_bits

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
    Supported: every write is one ``copy_`` of a row/column block of a 1-D or
    2-D checkpoint tensor into a param of the same dtype, with no post-load
    transform of the weights.
    """

    def __init__(self, model: Module, tensors: dict[str, tuple[list[int], str]]):
        device = next(model.parameters()).device
        params: dict[int, torch.Tensor] = {}
        for param in model.parameters():
            if param.storage_offset() == 0 and param.is_contiguous():
                params.setdefault(param.untyped_storage().data_ptr(), param)
        self.params: list[torch.Tensor] = list(params.values())
        param_index = {ptr: index for index, ptr in enumerate(params)}
        source_names: dict[int, str] = {}
        # name -> [(row0, rows, col0, cols, param index, offset, row stride, col stride)]
        records: dict[str, list[tuple[int, ...]]] = defaultdict(list)

        def record_copy_(destination: torch.Tensor, source: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
            name = source_names.get(source.untyped_storage().data_ptr())
            index = param_index.get(destination.untyped_storage().data_ptr())
            shape = tensors[name][0] if name is not None else None
            num_cols = shape[-1] if shape else 1
            source_2d = source.view(1, -1) if source.dim() == 1 else source
            destination_2d = destination.view(1, -1) if destination.dim() == 1 else destination
            if (
                name is None
                or index is None
                or len(shape) > 2
                or source.dtype != destination.dtype
                or source.shape != destination.shape
                or source.dim() > 2
                or source_2d.stride(-1) != 1
                or (source_2d.shape[0] > 1 and source_2d.stride(0) != num_cols)
            ):
                raise NotImplementedError("Sparse weight deltas need plain block copies from checkpoint tensors")
            row0, col0 = divmod(source.storage_offset(), num_cols)
            records[name].append(
                (row0, source_2d.shape[0], col0, source_2d.shape[1], index, destination.storage_offset())
                + tuple(destination_2d.stride())
            )
            return destination

        def placeholders():
            for name, (shape, dtype) in tensors.items():
                tensor = torch.empty(shape, dtype=getattr(torch, dtype), device=device)
                source_names[tensor.untyped_storage().data_ptr()] = name
                yield name, tensor

        original_copy = torch.Tensor.copy_
        torch.Tensor.copy_ = record_copy_
        try:
            model.load_weights(placeholders())  # type: ignore
        finally:
            torch.Tensor.copy_ = original_copy

        self.num_cols = {name: (shape[-1] if shape else 1) for name, (shape, _) in tensors.items()}
        self.records = dict(records)
        # [first, last + 1) flat checkpoint index each tensor's writes cover
        self.spans = {
            name: (
                min(row0 * self.num_cols[name] + col0 for row0, _, col0, *_ in writes),
                max((row0 + rows - 1) * self.num_cols[name] + col0 + cols for row0, rows, col0, cols, *_ in writes),
            )
            for name, writes in self.records.items()
        }

    @torch.no_grad()
    def apply_xor(
        self, entries: list[tuple[str, int]], indices: torch.Tensor, xor: torch.Tensor, chunk: int = 1 << 25
    ) -> None:
        """XOR ``xor`` (bits) into the params at the checkpoint ``indices`` of ``entries``."""
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
        for start in range(0, indices.numel(), chunk):
            index = indices[start : start + chunk]
            source = xor[start : start + chunk]
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
                self._scatter_xor(owner, target, source[hit])

    def _scatter_xor(self, owner: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> None:
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
            flat.index_put_((positions,), flat[positions] ^ source[begin:end])


@torch.no_grad()
def weight_checksums(model: Module, chunk: int = 1 << 24) -> dict[str, int]:
    """Position-weighted sum of every parameter's raw bits (exact, any dtype)."""
    sums = {}
    for name, param in model.named_parameters():
        bits = as_bits(param.data).reshape(-1)
        total = torch.zeros((), dtype=torch.int64, device=bits.device)
        for start in range(0, bits.numel(), chunk):
            part = bits[start : start + chunk].long()
            total += (part * torch.arange(start + 1, start + 1 + part.numel(), device=bits.device)).sum()
        sums[name] = total
    return {name: int(total) for name, total in sums.items()}


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
