import json
import warnings
from pathlib import Path
from typing import Literal, cast

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from prime_rl.trainer.lora import (
    clean_lora_state_dict,
)
from prime_rl.utils.logger import get_logger


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def load_state_dict(save_dir: Path) -> dict[str, Tensor]:
    """Load a state dict from a local directory with safetensor files."""
    safetensors_paths = list(save_dir.glob("*.safetensors"))
    if len(safetensors_paths) > 1:
        safetensors_paths.sort(key=lambda x: int(x.stem.split("-")[1].split("of")[0]))
    state_dict = {}
    for safetensor_path in safetensors_paths:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def save_state_dict(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    save_format: Literal["torch", "safetensors"] = "safetensors",
    save_sharded: bool = True,
    adapter: bool = False,
):
    """Save a state dict to a local directory in safetensors or torch format."""
    logger = get_logger()
    if adapter:
        weights_name = ADAPTER_SAFE_WEIGHTS_NAME if save_format == "safetensors" else ADAPTER_WEIGHTS_NAME
    else:
        weights_name = SAFE_WEIGHTS_NAME if save_format == "safetensors" else WEIGHTS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_sharded:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
        )
        if state_dict_split.is_sharded:
            filenames = state_dict_split.filename_to_tensors.keys()
            logger.debug(f"Saving sharded weights to {len(filenames)} files: ({', '.join(filenames)})")
        else:
            logger.debug(f"Saving unsharded weights to {weights_name}")

        # Save weights (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4252)
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                assert isinstance(state_dict[tensor], Tensor)
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]
            if save_format == "safetensors":
                save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
            else:
                torch.save(shard, save_dir / shard_file)
        del state_dict

        # Save index (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4301)
        if state_dict_split.is_sharded:
            index = {
                "metadata": {**state_dict_split.metadata},
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_format == "safetensors" else WEIGHTS_INDEX_NAME
            save_index_file = save_dir / save_index_file
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
    else:
        if save_format == "safetensors":
            save_file(state_dict, save_dir / weights_name, metadata={"format": "pt"})
        else:
            torch.save(state_dict, save_dir / weights_name)


def gather_weights_on_master(
    model: nn.Module, is_master: bool, dtype: torch.dtype = torch.bfloat16
) -> dict[str, Tensor]:
    """Gather distributed weights on CPU on master rank."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        cpu_state = {}
        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                # only gather after the downcast to dtype as it will be faster
                value = cast(DTensor, value.to(dtype)).full_tensor()

            if is_master:
                key = get_fqns(model, key)
                assert len(key) == 1
                key = next(iter(key))
                # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
                cpu_state[key] = value.to("cpu", non_blocking=False)
        torch.distributed.barrier()

    # Always clean up the state dict for HF compatibility
    if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in cpu_state.keys()):
        cpu_state = clean_lora_state_dict(cpu_state)

    return cpu_state


def get_adapter_state_dict(model: nn.Module, is_master: bool) -> dict[str, Tensor]:
    """Get adapter weights with clean keys for PEFT compatibility."""
    lora_state = {}

    for key, value in model.state_dict().items():
        param = dict(model.named_parameters()).get(key)
        if param is None or not param.requires_grad:
            continue

        if isinstance(value, DTensor):
            value = value.full_tensor()

        if is_master:
            clean_key = next(iter(get_fqns(model, key)))
            clean_key = clean_key.replace(".base_layer.", ".")

            # Add PEFT-expected prefix
            peft_key = f"base_model.model.{clean_key}"

            # Add .weight suffix for LoRA parameters if missing
            if ("lora_A" in peft_key or "lora_B" in peft_key) and not peft_key.endswith(".weight"):
                peft_key = f"{peft_key}.weight"

            lora_state[peft_key] = value.to("cpu", non_blocking=False)

    torch.distributed.barrier()
    return lora_state
