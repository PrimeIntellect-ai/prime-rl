import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers.utils import ADAPTER_SAFE_WEIGHTS_NAME

from prime_rl.configs.trainer import LoRAConfig, ModelConfig
from prime_rl.trainer.lora import apply_lora_to_model
from prime_rl.trainer.models.layers.lora import MultiLoRAModule
from prime_rl.trainer.model import DTYPE_MAP, configure_moe_ep_backend, get_model
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.utils.logger import get_logger


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    numel: int


@dataclass
class AdapterTemplate:
    model: nn.Module
    specs: list[TensorSpec]
    theta: torch.Tensor
    adapter_config: dict


def _init_lora_tensor(name: str, shape: torch.Size) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.float32)
    if "lora_B" in name:
        nn.init.zeros_(tensor)
    else:
        nn.init.kaiming_uniform_(tensor, a=5**0.5)
    return tensor


def flatten_state_specs(
    state_dict: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, list[TensorSpec]]:
    pieces: list[torch.Tensor] = []
    specs: list[TensorSpec] = []
    for name, tensor in state_dict.items():
        shape = torch.Size(tensor.shape)
        init = _init_lora_tensor(name, shape)
        pieces.append(init.reshape(-1))
        specs.append(TensorSpec(name=name, shape=shape, dtype=tensor.dtype, numel=init.numel()))
    if not pieces:
        raise RuntimeError("No LoRA tensors were found while building the ES adapter template.")
    theta = torch.cat(pieces).to(dtype=torch.float32)
    if device is not None:
        theta = theta.to(device=device)
    return theta, specs


def unflatten_state(
    flat: torch.Tensor,
    specs: list[TensorSpec],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    offset = 0
    cpu_flat = flat.detach().to("cpu")
    for spec in specs:
        tensor = cpu_flat[offset : offset + spec.numel].reshape(spec.shape).to(dtype=dtype).contiguous()
        state[spec.name] = tensor
        offset += spec.numel
    return state


def build_adapter_config(model: nn.Module, lora_config: LoRAConfig) -> dict:
    target_modules = set()
    modules_to_save = set()

    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
            target_modules.add(name.split(".")[-1])

    for name, param in model.named_parameters():
        if param.requires_grad and "lora_A" not in name and "lora_B" not in name:
            modules_to_save.add(name.rsplit(".", 1)[0].split(".")[-1])

    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model.config._name_or_path,
        "r": lora_config.rank,
        "lora_alpha": lora_config.alpha,
        "lora_dropout": lora_config.dropout,
        "bias": "none",
        "target_modules": sorted(target_modules),
        "modules_to_save": sorted(modules_to_save) if modules_to_save else None,
    }


def build_adapter_template(
    output_dir: Path,
    model_config: ModelConfig,
    *,
    device: torch.device | None = None,
) -> AdapterTemplate:
    if model_config.lora is None:
        raise ValueError("ES adapter template requires model.lora to be configured.")

    logger = get_logger()
    logger.info("Building ES LoRA adapter template on meta device")
    setup_multi_run_manager(output_dir, max_runs=1, device=torch.device("cpu"), lora_config=model_config.lora)
    model = get_model(model_config, device=torch.device("meta"), dtype=DTYPE_MAP[model_config.optimization_dtype])
    configure_moe_ep_backend(model, model_config)
    apply_lora_to_model(model, model_config.lora)
    adapter_config = build_adapter_config(model, model_config.lora)

    from prime_rl.trainer.runs import get_multi_run_manager

    manager = get_multi_run_manager()
    manager.reset_run_parameters(0)
    manager.scaling_factors[0] = model_config.lora.alpha / model_config.lora.rank
    state_dict = manager.get_state_dict_for_run(0)
    theta, specs = flatten_state_specs(state_dict, device=device)
    logger.info(
        f"ES LoRA search space has {theta.numel():,} parameters across {len(specs):,} tensors on {theta.device}"
    )
    return AdapterTemplate(model=model, specs=specs, theta=theta, adapter_config=adapter_config)


def write_adapter_from_theta(
    adapter_dir: Path,
    template: AdapterTemplate,
    theta: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    state = unflatten_state(theta, template.specs, dtype=dtype)
    save_file(state, adapter_dir / ADAPTER_SAFE_WEIGHTS_NAME, metadata={"format": "pt"})
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(template.adapter_config, f, indent=2)
