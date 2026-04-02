import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor, distribute_tensor

from prime_rl.configs.trainer import LoRAConfig
from prime_rl.trainer.models.layers.lora import MultiLoRALinear, MultiLoRAModule
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAGroupedExperts
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.utils.logger import get_logger

_MOE_LORA_KEY_RE = re.compile(
    r"(?P<prefix>.*\.experts)\.(?P<eid>\d+)\.(?P<proj>gate_proj|down_proj|up_proj)\.(?P<ab>lora_[AB])(?:\.(?:default|\d+))?(?:\.weight)?"
)
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")


def strip_lora_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip LoRA from the state dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        new_state_dict[key] = value
    return new_state_dict


def _strip_adapter_export_prefix(key: str) -> str:
    prefix = "base_model.model."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """Get a module by its fully qualified name."""
    parts = module_name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a module by its fully qualified name."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _has_regex_metacharacters(pattern: str) -> bool:
    """Check if a pattern contains regex metacharacters."""
    regex_metachars = {".", "*", "+", "?", "^", "$", "[", "]", "{", "}", "|", "(", ")", "\\"}
    return any(char in pattern for char in regex_metachars)


def _matches_pattern(name: str, pattern: str) -> bool:
    """Check if a name matches a pattern.

    For simple patterns (no regex metacharacters), checks if any component
    in the module path matches the pattern exactly. For regex patterns, uses
    re.search() to match anywhere in the name (mirroring PEFT behavior).

    This handles cases where Linear layers might be nested (e.g.,
    "model.layers.0.q_proj.linear") while still matching standard architectures
    where they're direct children (e.g., "model.layers.0.self_attn.q_proj").
    """
    if _has_regex_metacharacters(pattern):
        return re.search(pattern, name) is not None
    else:
        return pattern in name.split(".")


def _find_target_modules(model: nn.Module, target_patterns: List[str]) -> List[str]:
    """Find all module names that match any of the target patterns.

    Patterns can be simple module names (e.g., "q_proj") or regex patterns
    (e.g., r".*\\.q_proj$"). Simple names match any component in the module path.

    Supports both nn.Linear layers and GroupedExperts (MoE) modules.
    """
    target_modules = []

    for name, module in model.named_modules():
        # Check if module is Linear or GroupedExperts
        if not (isinstance(module, nn.Linear) or isinstance(module, GroupedExperts)):
            continue

        for pattern in target_patterns:
            if _matches_pattern(name, pattern):
                target_modules.append(name)
                break

    return target_modules


def _should_keep_trainable(param_name: str, modules_to_save_patterns: List[str]) -> bool:
    """Check if a parameter should remain fully trainable.

    Checks both the full parameter name and the parent module name against patterns.
    For example, for param "model.embed_tokens.weight", it checks both:
    - "model.embed_tokens.weight" (full parameter name)
    - "model.embed_tokens" (module name)

    Patterns can be simple module names (e.g., "embed_tokens") or regex patterns.
    """
    for pattern in modules_to_save_patterns:
        if _matches_pattern(param_name, pattern):
            return True

    module_name = param_name.rsplit(".", 1)[0] if "." in param_name else param_name
    for pattern in modules_to_save_patterns:
        if _matches_pattern(module_name, pattern):
            return True

    return False


def freeze_all_except_lora_and_specified(model: nn.Module, config: LoRAConfig) -> None:
    """
    Freeze all parameters except LoRA adapters and specified trainable modules.

    Args:
        model: The model to freeze parameters in
        config: LoRA configuration with modules_to_save patterns
    """
    for name, param in model.named_parameters():
        if any(lora_param in name for lora_param in ["lora_A", "lora_B"]):
            param.requires_grad = True
        elif _should_keep_trainable(name, config.modules_to_save):
            param.requires_grad = True
        else:
            param.requires_grad = False


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> None:
    """
    Apply LoRA to target modules in the model and freeze non-LoRA parameters.

    WARNING: This function modifies requires_grad on parameters. If using FSDP2,
    this MUST be called BEFORE setup_fsdp() to avoid dtensor/sharding issues.

    Args:
        model: The model to apply LoRA to
        config: LoRA configuration
    """
    logger = get_logger()
    n_loras = get_multi_run_manager().max_runs

    from torch.distributed.fsdp import FSDPModule

    if any(isinstance(m, FSDPModule) for m in model.modules()):
        logger.error(
            "Model is already wrapped with FSDP! LoRA must be applied BEFORE FSDP setup to avoid dtensor issues."
        )
        raise RuntimeError("Cannot apply LoRA to FSDP-wrapped model. Apply LoRA before setup_fsdp().")

    logger.debug(f"Applying LoRA to model: {model} for {config.target_modules}")
    target_modules = _find_target_modules(model, config.target_modules)
    logger.debug(
        f"Found {len(target_modules)} target modules for LoRA: {target_modules[:10]} ... {target_modules[-10:]}"
    )

    if not target_modules:
        logger.warning("No target modules found for LoRA. Check your target_modules patterns.")
        return

    for module_name in target_modules:
        base_module = _get_module_by_name(model, module_name)

        if isinstance(base_module, nn.Linear):
            lora_module = MultiLoRALinear(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=n_loras,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        # Handle GroupedExperts (MoE)
        elif isinstance(base_module, GroupedExperts):
            lora_module = MultiLoRAGroupedExperts(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=n_loras,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        else:
            logger.warning(
                f"Module {module_name} is type {type(base_module).__name__}, "
                f"expected nn.Linear or GroupedExperts. Skipping."
            )
            continue

        lora_module.register_with_runs(get_multi_run_manager(), module_name)
        _set_module_by_name(model, module_name, lora_module)

    freeze_all_except_lora_and_specified(model, config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_adapter_params = 0
    lora_adapted_params = 0
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
            adapter_params, adapted_params = module.get_lora_param_counts()
            lora_adapter_params += adapter_params
            lora_adapted_params += adapted_params

    fully_trainable = trainable_params - lora_adapter_params
    adapted_or_trainable = lora_adapted_params + fully_trainable

    logger.info(f"LoRA enabled: {lora_adapter_params:,} adapter params adapting {lora_adapted_params:,} base params")
    logger.info(f"LoRA: {fully_trainable:,} fully trainable parameters")
    logger.info(f"LoRA: {adapted_or_trainable:,} adapted or fully trainable out of {total_params:,} parameters")


def has_lora_layers(model: nn.Module) -> bool:
    """Check if model has LoRA layers."""
    for module in model.modules():
        if isinstance(module, MultiLoRAModule):
            return True
    return False


def clean_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove LoRA parameters and fix LoRA base layer key names for HF compatibility."""
    clean_state_dict = {}

    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue

        if ".base_layer." in key:
            new_key = key.replace(".base_layer.", ".")
            clean_state_dict[new_key] = value
        else:
            clean_state_dict[key] = value

    return clean_state_dict


def save_lora_config(model: nn.Module, save_path, rank: int, alpha: float, dropout: float) -> None:
    """
    Save LoRA configuration as JSON for adapter portability.

    Args:
        model: Model with LoRA layers to introspect
        save_path: Path object or string pointing to directory where adapter_config.json will be saved
        rank: LoRA rank
        alpha: LoRA alpha scaling parameter
        dropout: LoRA dropout rate
    """
    import json
    from pathlib import Path

    save_path = Path(save_path)

    # Extract actual target modules from the model
    target_modules = set()
    modules_to_save = set()

    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
            module_suffix = name.split(".")[-1]
            target_modules.add(module_suffix)

    for name, param in model.named_parameters():
        if param.requires_grad and "lora_A" not in name and "lora_B" not in name:
            module_name = name.rsplit(".", 1)[0].split(".")[-1]
            modules_to_save.add(module_name)

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model.config._name_or_path,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "target_modules": sorted(list(target_modules)),
        "modules_to_save": sorted(list(modules_to_save)) if modules_to_save else None,
    }

    config_path = save_path / ADAPTER_CONFIG_NAME
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)


def resolve_adapter_dir(init_adapter_path: Path) -> Path:
    """Resolve init_adapter_path to the adapter directory containing adapter_config.json."""
    if init_adapter_path.is_file():
        if init_adapter_path.name not in ADAPTER_WEIGHT_FILENAMES:
            raise ValueError(
                f"init_adapter_path file must be one of {ADAPTER_WEIGHT_FILENAMES}, got {init_adapter_path}"
            )
        adapter_dir = init_adapter_path.parent
    elif init_adapter_path.is_dir():
        adapter_dir = init_adapter_path
    else:
        raise ValueError(f"init_adapter_path does not exist: {init_adapter_path}")
    if not (adapter_dir / ADAPTER_CONFIG_NAME).exists():
        raise ValueError(f"Adapter directory is missing {ADAPTER_CONFIG_NAME}: {adapter_dir}")
    return adapter_dir


def _resolve_adapter_weights(adapter_dir: Path) -> Path:
    for name in ADAPTER_WEIGHT_FILENAMES:
        candidate = adapter_dir / name
        if candidate.exists():
            return candidate
    raise ValueError(f"No {ADAPTER_WEIGHT_FILENAMES} found in {adapter_dir}")


def _load_adapter_config(adapter_dir: Path, lora_config: LoRAConfig) -> dict:
    with open(adapter_dir / ADAPTER_CONFIG_NAME) as f:
        adapter_config = json.load(f)
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError(
            f"init_adapter_path only supports LoRA adapters, got peft_type={adapter_config.get('peft_type')}"
        )
    if adapter_config.get("r") != lora_config.rank:
        raise ValueError(f"init_adapter_path rank mismatch: expected {lora_config.rank}, got {adapter_config.get('r')}")
    if adapter_config.get("lora_alpha") != lora_config.alpha:
        raise ValueError(
            f"init_adapter_path alpha mismatch: expected {lora_config.alpha}, got {adapter_config.get('lora_alpha')}"
        )
    modules_to_save = adapter_config.get("modules_to_save")
    if modules_to_save not in (None, []):
        raise ValueError("init_adapter_path does not support adapters with modules_to_save")
    return adapter_config


def _normalize_lora_key(key: str) -> str:
    key = _strip_adapter_export_prefix(key)
    if key.endswith(".weight"):
        key = key[: -len(".weight")]
    key = re.sub(r"\.(lora_[AB])\.(default|\d+)", r".\1.0", key)
    key = re.sub(r"\.(lora_[AB])$", r".\1.0", key)
    return key


def _parse_moe_lora_key(key: str) -> tuple[str, int] | None:
    key = _strip_adapter_export_prefix(key)
    m = _MOE_LORA_KEY_RE.fullmatch(key)
    if m is None:
        return None
    proj_map = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    return f"{m.group('prefix')}.{proj_map[m.group('proj')]}_{m.group('ab')}.0", int(m.group("eid"))


def _set_adapter_idx_suffix(key: str, adapter_idx: int) -> str:
    if not key.endswith(".0"):
        raise ValueError(f"Expected adapter-slot suffix '.0' in key: {key}")
    return f"{key[:-2]}.{adapter_idx}"


def _is_model_lora_key_for_adapter(key: str, adapter_idx: int) -> bool:
    # Support the current standard LoRA and grouped-expert adapter suffixes.
    suffixes = (
        f"lora_A.{adapter_idx}",
        f"lora_B.{adapter_idx}",
        f"w1_lora_A.{adapter_idx}",
        f"w1_lora_B.{adapter_idx}",
        f"w2_lora_A.{adapter_idx}",
        f"w2_lora_B.{adapter_idx}",
        f"w3_lora_A.{adapter_idx}",
        f"w3_lora_B.{adapter_idx}",
    )
    return key.endswith(suffixes)


def _get_model_lora_state_keys(model_state: dict[str, torch.Tensor], adapter_idx: int) -> set[str]:
    return {key for key in model_state if _is_model_lora_key_for_adapter(key, adapter_idx)}


def _get_load_path_label(mapped_keys: dict[str, torch.Tensor]) -> str:
    has_moe = any(".experts." in key for key in mapped_keys)
    has_dense = any(".lora_" in key and ".experts." not in key for key in mapped_keys)
    if has_dense and has_moe:
        return "mixed"
    if has_moe:
        return "moe"
    return "normal"


def _raise_lora_key_mismatch(
    *,
    init_adapter_path: Path,
    adapter_idx: int,
    expected_keys: set[str],
    actual_keys: set[str],
    load_path: str,
) -> None:
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    matched = len(expected_keys & actual_keys)
    raise ValueError(
        "LoRA key mismatch. "
        f"adapter_path={init_adapter_path}, adapter_idx={adapter_idx}, load_path={load_path}, "
        f"loaded_keys={len(actual_keys)}, matched_model_keys={matched}/{len(expected_keys)}, "
        f"missing_sample={missing[:5]}, unexpected_sample={unexpected[:5]}"
    )


@dataclass(frozen=True)
class PreparedInitAdapter:
    init_adapter_path: Path
    slot0_tensors: dict[str, torch.Tensor]
    load_path: str

    def apply_to_model(self, model: nn.Module, adapter_idx: int = 0) -> None:
        model_state = model.state_dict()
        expected_keys = _get_model_lora_state_keys(model_state, adapter_idx)
        mapped = {
            _set_adapter_idx_suffix(key, adapter_idx): value
            for key, value in self.slot0_tensors.items()
        }

        if set(mapped) != expected_keys:
            _raise_lora_key_mismatch(
                init_adapter_path=self.init_adapter_path,
                adapter_idx=adapter_idx,
                expected_keys=expected_keys,
                actual_keys=set(mapped),
                load_path=self.load_path,
            )

        aligned = {}
        for key, value in mapped.items():
            target = model_state[key]
            if value.shape != target.shape:
                raise ValueError(
                    f"LoRA tensor shape mismatch for {key}: expected {target.shape}, got {value.shape}"
                )
            value = value.to(dtype=target.dtype)
            if isinstance(target, DTensor):
                aligned[key] = distribute_tensor(
                    value.to(device=target.device),
                    target.device_mesh,
                    target.placements,
                )
            else:
                aligned[key] = value.to(device=target.device)

        model.load_state_dict(aligned, strict=False)

    def register_creation_hook(self, model: nn.Module) -> None:
        if getattr(model, "_prime_init_adapter_creation_hook_registered", False):
            return

        def _apply_prepared_init_adapter(idx: int, _run_id: str) -> None:
            self.apply_to_model(model, adapter_idx=idx)

        get_multi_run_manager().register_creation_hook(_apply_prepared_init_adapter)
        setattr(model, "_prime_init_adapter_creation_hook_registered", True)


def prepare_init_adapter(model: nn.Module, init_adapter_path: Path, lora_config: LoRAConfig) -> PreparedInitAdapter:
    adapter_dir = resolve_adapter_dir(init_adapter_path)
    _load_adapter_config(adapter_dir, lora_config)
    weights_path = _resolve_adapter_weights(adapter_dir)
    raw = (
        load_file(str(weights_path), device="cpu")
        if weights_path.suffix == ".safetensors"
        else torch.load(weights_path, map_location="cpu", weights_only=True)
    )

    mapped: dict[str, torch.Tensor] = {}
    moe_parts: dict[str, dict[int, torch.Tensor]] = {}
    for key, value in raw.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        moe = _parse_moe_lora_key(key)
        if moe is not None:
            target_key, expert_id = moe
            moe_parts.setdefault(target_key, {})[expert_id] = value
        else:
            mapped[_normalize_lora_key(key)] = value

    for target_key, parts in moe_parts.items():
        count = len(parts)
        if set(parts) != set(range(count)):
            raise ValueError(f"Missing MoE expert slices for {target_key}")
        mapped[target_key] = torch.stack([parts[i] for i in range(count)], dim=0)

    if not mapped:
        raise ValueError("No LoRA tensors found in init adapter")

    model_state = model.state_dict()
    expected_keys = _get_model_lora_state_keys(model_state, 0)
    load_path = _get_load_path_label(mapped)
    if set(mapped) != expected_keys:
        _raise_lora_key_mismatch(
            init_adapter_path=init_adapter_path,
            adapter_idx=0,
            expected_keys=expected_keys,
            actual_keys=set(mapped),
            load_path=load_path,
        )

    for key, value in mapped.items():
        target = model_state[key]
        if value.shape != target.shape:
            raise ValueError(f"LoRA tensor shape mismatch for {key}: expected {target.shape}, got {value.shape}")

    return PreparedInitAdapter(
        init_adapter_path=init_adapter_path,
        slot0_tensors={key: value.detach().to("cpu", non_blocking=False) for key, value in mapped.items()},
        load_path=load_path,
    )
