from collections.abc import Generator, Iterable

import torch
import torch.nn as nn

KV_PREFIX_KEY_SUFFIX = ".kv_prefix_key"
KV_PREFIX_VALUE_SUFFIX = ".kv_prefix_value"


def is_kv_prefix_weight_name(name: str) -> bool:
    return name.endswith(KV_PREFIX_KEY_SUFFIX) or name.endswith(KV_PREFIX_VALUE_SUFFIX)


def split_kv_prefix_state_dict(
    weights_iterator: Iterable[tuple[str, torch.Tensor]],
) -> tuple[Generator[tuple[str, torch.Tensor], None, None], dict[str, torch.Tensor]]:
    kv_prefix_state_dict: dict[str, torch.Tensor] = {}

    def _iterator() -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, tensor in weights_iterator:
            if is_kv_prefix_weight_name(name):
                kv_prefix_state_dict[name] = tensor
                continue
            yield name, tensor

    return _iterator(), kv_prefix_state_dict


def _iter_attention_layers(model: nn.Module) -> Generator[tuple[str, nn.Module], None, None]:
    for _, module in model.named_modules():
        layer_name = getattr(module, "layer_name", None)
        if not isinstance(layer_name, str):
            continue
        if not hasattr(module, "num_kv_heads") or not hasattr(module, "head_size"):
            continue
        yield layer_name, module


def clear_kv_prefix_state(model: nn.Module) -> None:
    for _, layer in _iter_attention_layers(model):
        layer._prime_kv_prefix_key = None
        layer._prime_kv_prefix_value = None
        layer._prime_kv_prefix_num_tokens = 0


def _split_kv_prefix_key(name: str) -> tuple[str, str] | None:
    if name.endswith(KV_PREFIX_KEY_SUFFIX):
        return name[: -len(KV_PREFIX_KEY_SUFFIX)], "key"
    if name.endswith(KV_PREFIX_VALUE_SUFFIX):
        return name[: -len(KV_PREFIX_VALUE_SUFFIX)], "value"
    return None


def _resolve_layer_name(base_name: str, available_layer_names: set[str]) -> str:
    if base_name in available_layer_names:
        return base_name

    with_attn = f"{base_name}.attn"
    if with_attn in available_layer_names:
        return with_attn

    if base_name.endswith(".attn"):
        without_attn = base_name[: -len(".attn")]
        if without_attn in available_layer_names:
            return without_attn

    raise ValueError(f"No matching vLLM attention layer for KV-prefix tensor base '{base_name}'")


def _reshape_prefix_tensor_for_layer(layer: nn.Module, tensor: torch.Tensor, kind: str) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"KV-prefix {kind} tensor must be rank-3, got shape {tuple(tensor.shape)}")

    num_kv_heads = int(layer.num_kv_heads)
    head_size = int(layer.head_size)

    if tensor.shape[0] == num_kv_heads and tensor.shape[2] == head_size:
        # Trainer layout: [num_kv_heads, num_prefix_tokens, head_size]
        return tensor.transpose(0, 1).contiguous()

    if tensor.shape[1] == num_kv_heads and tensor.shape[2] == head_size:
        # Already in inference layout: [num_prefix_tokens, num_kv_heads, head_size]
        return tensor.contiguous()

    raise ValueError(
        f"KV-prefix {kind} tensor shape {tuple(tensor.shape)} is incompatible with "
        f"layer (num_kv_heads={num_kv_heads}, head_size={head_size})"
    )


def _validate_attention_backend(layer: nn.Module, layer_name: str) -> None:
    impl = getattr(layer, "impl", None)
    if impl is None:
        return
    if "flash_attn" not in type(impl).__module__:
        raise ValueError(
            f"KV-prefix inference currently requires FlashAttention backend, but layer '{layer_name}' uses "
            f"{type(impl).__module__}.{type(impl).__name__}"
        )


def apply_kv_prefix_state_dict(model: nn.Module, kv_prefix_state_dict: dict[str, torch.Tensor]) -> int:
    clear_kv_prefix_state(model)
    if len(kv_prefix_state_dict) == 0:
        return 0

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in kv_prefix_state_dict.items():
        split = _split_kv_prefix_key(name)
        if split is None:
            continue
        base_name, kind = split
        grouped.setdefault(base_name, {})[kind] = tensor

    if len(grouped) == 0:
        return 0

    layers = dict(_iter_attention_layers(model))
    available_layer_names = set(layers.keys())

    applied_layers = 0
    for base_name, tensors in grouped.items():
        if "key" not in tensors or "value" not in tensors:
            raise ValueError(f"KV-prefix tensors for '{base_name}' must include both key and value")

        layer_name = _resolve_layer_name(base_name, available_layer_names)
        layer = layers[layer_name]
        _validate_attention_backend(layer, layer_name)

        key = _reshape_prefix_tensor_for_layer(layer, tensors["key"], "key")
        value = _reshape_prefix_tensor_for_layer(layer, tensors["value"], "value")

        if key.shape != value.shape:
            raise ValueError(
                f"KV-prefix key/value shapes do not match for layer '{layer_name}': "
                f"{tuple(key.shape)} vs {tuple(value.shape)}"
            )

        prefix_tokens = int(key.shape[0])
        if prefix_tokens < 1:
            raise ValueError(f"KV-prefix length must be >= 1 for layer '{layer_name}'")

        device = layer._k_scale.device
        layer._prime_kv_prefix_key = key.to(device=device, non_blocking=False)
        layer._prime_kv_prefix_value = value.to(device=device, non_blocking=False)
        layer._prime_kv_prefix_num_tokens = prefix_tokens
        applied_layers += 1

    return applied_layers


def get_layer_kv_prefix(layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    key = getattr(layer, "_prime_kv_prefix_key", None)
    value = getattr(layer, "_prime_kv_prefix_value", None)
    num_tokens = int(getattr(layer, "_prime_kv_prefix_num_tokens", 0))

    if key is None or value is None or num_tokens == 0:
        return None

    return key, value, num_tokens
