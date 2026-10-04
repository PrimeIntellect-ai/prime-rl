import time
from typing import Iterable

import torch
from safetensors import safe_open
from torch.nn import Module
from vllm.config import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_reload, initialize_layerwise_reload

from prime_rl.utils.delta import (
    DELTA_INDEX_SUFFIX,
    DELTA_LAYOUT_LOGICAL,
    DELTA_METADATA_FORMAT_V1,
    DELTA_METADATA_FORMAT_V2,
    DELTA_VALUE_SUFFIX,
    decode_sparse_indices,
    is_sparse_delta_store,
    is_streaming_delta_file,
    iter_streaming_delta_records,
    sparse_delta_format,
    sparse_delta_layout,
    sparse_delta_shapes,
)

from .weight_layout import DenseVllmModelLayoutAdapter, LogicalSparseUpdate

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


@torch.no_grad()
def load_sparse_delta_weights(model: Module, delta_path: str, scale_factor: float = 1.0) -> None:
    """Apply a sparse delta artifact to the current model weights in-place."""
    params = dict(model.named_parameters())
    start_time = time.perf_counter()
    updated = 0
    delta_values = 0
    full_tensor_updates = 0

    if is_streaming_delta_file(delta_path):
        adapter = DenseVllmModelLayoutAdapter(model)
        seen: set[str] = set()
        for record in iter_streaming_delta_records(delta_path):
            if record.name in seen:
                raise ValueError(f"duplicate sparse delta tensor: {record.name}")
            seen.add(record.name)

            if record.format_version == 2:
                if record.shape is None:
                    raise ValueError(f"logical streaming delta has no shape for {record.name}")
                local = adapter.to_local(
                    LogicalSparseUpdate(
                        name=record.name,
                        shape=record.shape,
                        indices=decode_sparse_indices(record.encoded_indices, record.values.numel()),
                        values=record.values,
                    )
                )
                full_tensor = _apply_decoded_sparse_delta_tensor(
                    params[local.name],
                    local.name,
                    local.indices,
                    local.values,
                    scale_factor,
                )
            else:
                param = params.get(record.name)
                if param is None:
                    raise ValueError(f"delta parameter is not present in the vLLM model: {record.name}")
                full_tensor = _apply_sparse_delta_tensor(
                    param,
                    record.name,
                    record.encoded_indices,
                    record.values,
                    scale_factor,
                )
            updated += 1
            delta_values += record.values.numel()
            full_tensor_updates += int(full_tensor)
        _log_sparse_delta_apply(delta_path, start_time, updated, delta_values, full_tensor_updates)
        return

    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        delta_format = sparse_delta_format(delta)
        if delta_format not in (None, DELTA_METADATA_FORMAT_V1, DELTA_METADATA_FORMAT_V2):
            raise ValueError(f"unsupported sparse delta format: {delta_format}")
        if delta_format == DELTA_METADATA_FORMAT_V2 and sparse_delta_layout(delta) != DELTA_LAYOUT_LOGICAL:
            raise ValueError(f"unsupported logical sparse delta layout: {sparse_delta_layout(delta)}")
        delta_keys = set(delta.keys())
        idx_names = {key[: -len(DELTA_INDEX_SUFFIX)] for key in delta_keys if key.endswith(DELTA_INDEX_SUFFIX)}
        val_names = {key[: -len(DELTA_VALUE_SUFFIX)] for key in delta_keys if key.endswith(DELTA_VALUE_SUFFIX)}

        if not idx_names and not val_names and not is_sparse_delta_store(delta):
            raise ValueError(f"{delta_path} does not look like a sparse delta file")
        if not idx_names and not val_names:
            logger.info(f"Sparse delta {delta_path} has no changed tensors")
            return
        if idx_names != val_names:
            raise ValueError(
                f"delta index/value names mismatch: idx-only={idx_names - val_names}, val-only={val_names - idx_names}"
            )

        logger.info(f"Applying sparse delta from {delta_path} ({len(idx_names)} tensors)")
        if delta_format == DELTA_METADATA_FORMAT_V2:
            shapes = sparse_delta_shapes(delta)
            if set(shapes) != idx_names:
                raise ValueError(
                    f"logical delta shape/name mismatch: "
                    f"missing={idx_names - set(shapes)}, unexpected={set(shapes) - idx_names}"
                )
            adapter = DenseVllmModelLayoutAdapter(model)
            for name in sorted(idx_names):
                values_cpu = delta.get_tensor(f"{name}{DELTA_VALUE_SUFFIX}").reshape(-1)
                local = adapter.to_local(
                    LogicalSparseUpdate(
                        name=name,
                        shape=shapes[name],
                        indices=decode_sparse_indices(
                            delta.get_tensor(f"{name}{DELTA_INDEX_SUFFIX}"), values_cpu.numel()
                        ),
                        values=values_cpu,
                    )
                )
                full_tensor = _apply_decoded_sparse_delta_tensor(
                    params[local.name],
                    local.name,
                    local.indices,
                    local.values,
                    scale_factor,
                )
                updated += 1
                delta_values += values_cpu.numel()
                full_tensor_updates += int(full_tensor)
        else:
            missing = sorted(idx_names - set(params))
            if missing:
                raise ValueError(
                    f"delta contains {len(missing)} parameter(s) not present in the vLLM model: {missing[:10]}"
                )
            for name in sorted(idx_names):
                param = params[name]
                values_cpu = delta.get_tensor(f"{name}{DELTA_VALUE_SUFFIX}").reshape(-1)
                full_tensor = _apply_sparse_delta_tensor(
                    param,
                    name,
                    delta.get_tensor(f"{name}{DELTA_INDEX_SUFFIX}"),
                    values_cpu,
                    scale_factor,
                )
                updated += 1
                delta_values += values_cpu.numel()
                full_tensor_updates += int(full_tensor)

    _log_sparse_delta_apply(delta_path, start_time, updated, delta_values, full_tensor_updates)


def _apply_sparse_delta_tensor(
    param: torch.nn.Parameter,
    name: str,
    encoded_indices: torch.Tensor,
    values_cpu: torch.Tensor,
    scale_factor: float,
) -> bool:
    values_cpu = values_cpu.reshape(-1)
    indices = decode_sparse_indices(encoded_indices, values_cpu.numel())
    return _apply_decoded_sparse_delta_tensor(param, name, indices, values_cpu, scale_factor)


def _apply_decoded_sparse_delta_tensor(
    param: torch.nn.Parameter,
    name: str,
    indices: torch.Tensor,
    values_cpu: torch.Tensor,
    scale_factor: float,
) -> bool:
    if not param.is_contiguous():
        raise ValueError(f"sparse delta requires a contiguous parameter: {name}")
    flat = param.data.view(-1)
    values_cpu = values_cpu.reshape(-1)
    if indices.numel() != values_cpu.numel():
        raise ValueError(
            f"sparse delta index/value length mismatch for {name}: {indices.numel()} vs {values_cpu.numel()}"
        )
    if indices.numel() == 0:
        return False
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= flat.numel():
        raise ValueError(f"sparse delta index out of range for {name}")
    if indices.numel() > 1 and bool(torch.any(indices[1:] <= indices[:-1])):
        raise ValueError(f"sparse delta indices must be strictly increasing for {name}")

    values = values_cpu.to(device=flat.device)
    if scale_factor != 1.0:
        values = values * scale_factor
    full_tensor = _indices_cover_flat_tensor(indices, flat.numel())
    if full_tensor and values.dtype == flat.dtype:
        flat.add_(values)
    else:
        indices = indices.to(device=flat.device)
        updated_values = flat.index_select(0, indices).to(torch.promote_types(flat.dtype, values.dtype)) + values
        flat.index_copy_(0, indices, updated_values.to(flat.dtype))
    return full_tensor


def _log_sparse_delta_apply(
    delta_path: str,
    start_time: float,
    updated: int,
    delta_values: int,
    full_tensor_updates: int,
) -> None:
    logger.info(
        f"Applied sparse delta {delta_path} to {updated} tensors ({delta_values} values, "
        f"{full_tensor_updates} full-tensor updates) in {time.perf_counter() - start_time:.2f}s"
    )


def _indices_cover_flat_tensor(indices: torch.Tensor, flat_numel: int) -> bool:
    return (
        indices.numel() == flat_numel
        and flat_numel > 0
        and int(indices[0].item()) == 0
        and int(indices[-1].item()) == flat_numel - 1
    )


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
