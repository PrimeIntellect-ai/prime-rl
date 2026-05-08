from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from prime_rl.configs.trainer import LoRAConfig, ModelConfig
from prime_rl.trainer.lora import apply_lora_to_model, save_lora_config
from prime_rl.trainer.model import DTYPE_MAP, configure_moe_ep_backend, get_model
from prime_rl.trainer.runs import setup_multi_run_manager
from prime_rl.trainer.weights import save_state_dict
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


def _init_lora_tensor(name: str, shape: torch.Size) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.float32)
    if "lora_B" in name:
        nn.init.zeros_(tensor)
    else:
        nn.init.kaiming_uniform_(tensor, a=5**0.5)
    return tensor


def flatten_state_specs(state_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[TensorSpec]]:
    pieces: list[torch.Tensor] = []
    specs: list[TensorSpec] = []
    for name, tensor in state_dict.items():
        shape = torch.Size(tensor.shape)
        init = _init_lora_tensor(name, shape)
        pieces.append(init.reshape(-1))
        specs.append(TensorSpec(name=name, shape=shape, dtype=tensor.dtype, numel=init.numel()))
    if not pieces:
        raise RuntimeError("No LoRA tensors were found while building the ES adapter template.")
    return torch.cat(pieces).to(torch.float32), specs


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


def build_adapter_template(output_dir: Path, model_config: ModelConfig) -> AdapterTemplate:
    if model_config.lora is None:
        raise ValueError("ES adapter template requires model.lora to be configured.")

    logger = get_logger()
    logger.info("Building ES LoRA adapter template on meta device")
    setup_multi_run_manager(output_dir, max_runs=1, device=torch.device("cpu"), lora_config=model_config.lora)
    model = get_model(model_config, device=torch.device("meta"), dtype=DTYPE_MAP[model_config.optimization_dtype])
    configure_moe_ep_backend(model, model_config)
    apply_lora_to_model(model, model_config.lora)

    from prime_rl.trainer.runs import get_multi_run_manager

    manager = get_multi_run_manager()
    manager.reset_run_parameters(0)
    manager.scaling_factors[0] = model_config.lora.alpha / model_config.lora.rank
    state_dict = manager.get_state_dict_for_run(0)
    theta, specs = flatten_state_specs(state_dict)
    logger.info(f"ES LoRA search space has {theta.numel():,} parameters across {len(specs):,} tensors")
    return AdapterTemplate(model=model, specs=specs, theta=theta)


def write_adapter_from_theta(
    adapter_dir: Path,
    template: AdapterTemplate,
    lora_config: LoRAConfig,
    theta: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    state = unflatten_state(theta, template.specs, dtype=dtype)
    save_state_dict(state, adapter_dir, save_format="safetensors", save_sharded=False, adapter=True)
    save_lora_config(
        template.model, adapter_dir, rank=lora_config.rank, alpha=lora_config.alpha, dropout=lora_config.dropout
    )
