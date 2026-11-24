"""Torch-optional loop-key helpers and exact sequential unrolling for looped NemotronH checkpoints."""

from __future__ import annotations

from typing import Any


def is_loop_param_name(name: str) -> bool:
    """Return True if a parameter or state-dict key belongs to a recurrent loop adapter."""
    return ".loop_" in f".{name}"


def strip_loop_from_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``state_dict`` without loop-adapter keys (for base HF loading)."""
    return {k: v for k, v in state_dict.items() if not is_loop_param_name(k)}


def resolve_loop_bounds(loop_config: dict[str, Any], num_hidden_layers: int) -> tuple[int, int]:
    """Resolve ``(start_layer, end_layer)`` for a ``loop_config`` dict."""
    start = int(loop_config["start_layer"])
    raw_end = loop_config.get("end_layer")
    if raw_end is None:
        end = num_hidden_layers - 4
    else:
        raw_end = int(raw_end)
        end = num_hidden_layers + raw_end if raw_end < 0 else raw_end
    if not (0 <= start < end <= num_hidden_layers):
        raise ValueError(f"Invalid loop bounds [{start}, {end}) for num_hidden_layers={num_hidden_layers}")
    return start, end


def _to_f32(arr: Any) -> Any:
    return arr.float() if hasattr(arr, "float") else arr.astype("float32")


def _cast_like(arr: Any, ref: Any) -> Any:
    return arr.to(dtype=ref.dtype) if hasattr(arr, "to") else arr.astype(ref.dtype)


def _fold_row_scale(arr: Any, scale: Any, axis: int = 0) -> Any:
    """Multiply ``arr`` by 1D ``scale`` along ``axis`` in fp32 and cast back to ``arr``'s dtype."""
    arr_f = _to_f32(arr)
    s_f = _to_f32(scale)
    shape = [1] * arr_f.ndim
    shape[axis] = s_f.shape[0]
    return _cast_like(arr_f * s_f.reshape(shape), arr)


def unroll_looped_state_dict(
    state_dict: dict[str, Any],
    config_dict: dict[str, Any],
    loop_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unroll a Prime-format looped NemotronH state dict into an exact sequential NemotronH state dict.

    Merges per-loop LoRA deltas, swaps in per-loop norms/routers, and folds ``loop_gammas`` row-wise
    into each looped layer's output projection(s). Works on both ``torch.Tensor`` and ``numpy.ndarray``.
    """
    pattern = config_dict.get("hybrid_override_pattern")
    if not pattern:
        raise ValueError("config_dict must define hybrid_override_pattern to unroll NemotronH")
    num_layers = len(pattern)
    num_loops = int(loop_config["num_loops"])
    start, end = resolve_loop_bounds(loop_config, num_layers)
    rank = int(loop_config.get("lora_rank", 0))
    alpha = float(loop_config.get("lora_alpha", 32.0))
    scaling = (alpha / rank) if rank > 0 else 0.0

    out_sd: dict[str, Any] = {}
    for k, v in state_dict.items():
        if not k.startswith("model.layers."):
            out_sd[k] = v

    layer_keys: list[dict[str, Any]] = [{} for _ in range(num_layers)]
    for k, v in state_dict.items():
        if not k.startswith("model.layers."):
            continue
        rest = k[len("model.layers.") :]
        idx_str, suffix = rest.split(".", 1)
        layer_keys[int(idx_str)][suffix] = v

    def emit_layer(src_idx: int, dst_idx: int, loop_idx: int) -> None:
        src = layer_keys[src_idx]
        dst_prefix = f"model.layers.{dst_idx}."
        base_items = {suf: val for suf, val in src.items() if not is_loop_param_name(suf)}
        if loop_idx == 0:
            for suf, val in base_items.items():
                out_sd[dst_prefix + suf] = val
            return

        extra = loop_idx - 1
        merged = dict(base_items)

        ln_key = f"loop_norms.{extra}.weight"
        if ln_key in src:
            merged["norm.weight"] = src[ln_key]

        rg_key = f"loop_routers.{extra}.gate.weight"
        if rg_key in src:
            merged["mlp.router.gate.weight"] = src[rg_key]
        rb_key = f"loop_routers.{extra}.selection_bias"
        if rb_key in src:
            merged["mlp.router.selection_bias"] = src[rb_key]

        if rank > 0:
            suffix_a = f".loop_lora_A.{extra}"
            for k_a, val_a in src.items():
                if not k_a.endswith(suffix_a):
                    continue
                proj_prefix = k_a[: -len(suffix_a)]
                val_b = src[f"{proj_prefix}.loop_lora_B.{extra}"]
                w_key = f"{proj_prefix}.weight"
                w_base = merged[w_key]
                delta = _to_f32(val_b) @ _to_f32(val_a)
                merged[w_key] = _cast_like(_to_f32(w_base) + scaling * delta, w_base)

        gamma_key = f"loop_gammas.{extra}"
        if gamma_key in src:
            gamma = src[gamma_key]
            layer_char = pattern[src_idx]
            if layer_char == "*":
                merged["self_attn.o_proj.weight"] = _fold_row_scale(merged["self_attn.o_proj.weight"], gamma, axis=0)
                if "self_attn.o_proj.bias" in merged:
                    merged["self_attn.o_proj.bias"] = _fold_row_scale(merged["self_attn.o_proj.bias"], gamma, axis=0)
            elif layer_char == "M":
                merged["mamba.out_proj.weight"] = _fold_row_scale(merged["mamba.out_proj.weight"], gamma, axis=0)
                if "mamba.out_proj.bias" in merged:
                    merged["mamba.out_proj.bias"] = _fold_row_scale(merged["mamba.out_proj.bias"], gamma, axis=0)
            elif layer_char == "E":
                if "mlp.fc2_latent_proj.weight" in merged:
                    merged["mlp.fc2_latent_proj.weight"] = _fold_row_scale(
                        merged["mlp.fc2_latent_proj.weight"], gamma, axis=0
                    )
                    if "mlp.fc2_latent_proj.bias" in merged:
                        merged["mlp.fc2_latent_proj.bias"] = _fold_row_scale(
                            merged["mlp.fc2_latent_proj.bias"], gamma, axis=0
                        )
                elif "mlp.experts.down_proj" in merged:
                    merged["mlp.experts.down_proj"] = _fold_row_scale(merged["mlp.experts.down_proj"], gamma, axis=1)
                if "mlp.shared_expert.down_proj.weight" in merged:
                    merged["mlp.shared_expert.down_proj.weight"] = _fold_row_scale(
                        merged["mlp.shared_expert.down_proj.weight"], gamma, axis=0
                    )
                    if "mlp.shared_expert.down_proj.bias" in merged:
                        merged["mlp.shared_expert.down_proj.bias"] = _fold_row_scale(
                            merged["mlp.shared_expert.down_proj.bias"], gamma, axis=0
                        )
            elif layer_char == "-":
                merged["mlp.down_proj.weight"] = _fold_row_scale(merged["mlp.down_proj.weight"], gamma, axis=0)
                if "mlp.down_proj.bias" in merged:
                    merged["mlp.down_proj.bias"] = _fold_row_scale(merged["mlp.down_proj.bias"], gamma, axis=0)

        for suf, val in merged.items():
            out_sd[dst_prefix + suf] = val

    dst_idx = 0
    for src_idx in range(0, start):
        emit_layer(src_idx, dst_idx, loop_idx=0)
        dst_idx += 1
    for loop_idx in range(num_loops):
        for src_idx in range(start, end):
            emit_layer(src_idx, dst_idx, loop_idx=loop_idx)
            dst_idx += 1
    for src_idx in range(end, num_layers):
        emit_layer(src_idx, dst_idx, loop_idx=0)
        dst_idx += 1

    new_pattern = pattern[:start] + pattern[start:end] * num_loops + pattern[end:]
    new_cfg = dict(config_dict)
    new_cfg["hybrid_override_pattern"] = new_pattern
    new_cfg["num_hidden_layers"] = len(new_pattern)
    new_cfg.pop("loop_config", None)
    return out_sd, new_cfg
