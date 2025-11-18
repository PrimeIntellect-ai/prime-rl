import json
import shutil
import threading
import time
import warnings
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from prime_rl.trainer.config import CheckpointConfig, LoRAConfig
from prime_rl.trainer.lora import (
    clean_lora_state_dict,
    has_lora_layers,
    merge_lora_weights_inplace,
    restore_lora_weights_inplace,
    save_lora_config,
)
from prime_rl.trainer.rl.config import WeightCheckpointConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_step_path, get_weight_ckpt_model_path, get_weights_dir


def has_hf_moe_layers(state_dict: dict[str, Tensor]) -> bool:
    """Whether the model contains MoE layers in HF format."""
    return any("mlp.experts.1.up_proj" in module_name for module_name in state_dict.keys())


def has_tt_moe_layers(state_dict: dict[str, Tensor]) -> bool:
    """Whether the model contains MoE layers in TT format."""
    return any("mlp.experts.w1" in module_name for module_name in state_dict.keys())


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from HF to TT format in-place."""
    num_layers = len(list(i for i in state_dict.keys() if "mlp.gate.weight" in i))
    num_experts = len(list(i for i in state_dict.keys() if "model.layers.2.mlp.experts" in i)) // 3

    for i in range(1, num_layers + 1):
        state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[f"model.layers.{i}.mlp.gate.weight"]
        del state_dict[f"model.layers.{i}.mlp.gate.weight"]

        dim, moe_dim = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].shape
        w1 = torch.empty(
            (num_experts, moe_dim, dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.1.down_proj.weight"].dtype
        )  # Gate
        w2 = torch.empty(
            (num_experts, dim, moe_dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.1.down_proj.weight"].dtype
        )  # Down
        w3 = torch.empty(
            (num_experts, moe_dim, dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.1.down_proj.weight"].dtype
        )  # Up
        for j in range(num_experts):
            w1[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"])
            w2[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"])
            w3[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"])

            del state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"]
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"]
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"]

        state_dict[f"model.layers.{i}.mlp.experts.w1"] = w1
        state_dict[f"model.layers.{i}.mlp.experts.w2"] = w2
        state_dict[f"model.layers.{i}.mlp.experts.w3"] = w3

        state_dict[f"model.layers.{i}.mlp.shared_expert.w1"] = state_dict[
            f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_expert.w2"] = state_dict[
            f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_expert.w3"] = state_dict[
            f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
        ]

        del state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"]
        del state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"]
        del state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"]

        state_dict[f"model.layers.{i}.mlp.expert_bias"] = state_dict[
            f"model.layers.{i}.mlp.gate.e_score_correction_bias"
        ]
        del state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    """Convert a layer from TT to HF format in-place."""

    i = layer_index

    # Load balancing terms
    if f"model.layers.{i}.mlp.expert_bias" in state_dict:
        state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = state_dict[
            f"model.layers.{i}.mlp.expert_bias"
        ]
        del state_dict[f"model.layers.{i}.mlp.expert_bias"]
    if f"model.layers.{i}.mlp.tokens_per_expert" in state_dict:
        del state_dict[f"model.layers.{i}.mlp.tokens_per_expert"]

    # Shared experts
    if f"model.layers.{i}.mlp.shared_expert.w1" in state_dict:
        state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w1"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w2"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w3"
        ]

        if state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"].shape[0] == 1:
            state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
            ][0]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w1"]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w2"]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w3"]

        # Gate / Router
        state_dict[f"model.layers.{i}.mlp.gate.weight"] = state_dict[f"model.layers.{i}.mlp.router.gate.weight"]
        del state_dict[f"model.layers.{i}.mlp.router.gate.weight"]

        # Routed experts
        num_experts, moe_dim, dim = state_dict[f"model.layers.{i}.mlp.experts.w1"].shape
        for j in range(num_experts):
            state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.experts.w1"
            ][j]
            state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.experts.w2"
            ][j]
            state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.experts.w3"
            ][j]
        del state_dict[f"model.layers.{i}.mlp.experts.w1"]
        del state_dict[f"model.layers.{i}.mlp.experts.w2"]
        del state_dict[f"model.layers.{i}.mlp.experts.w3"]


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from TT to HF format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(1, num_layers + 1):
        # todo(sami): delete this after testing that it never called
        # if not f"model.layers.{i}.mlp.router.gate.weight" in state_dict:
        #     continue  # Not a TT-MoE layer

        convert_tt_layer_to_hf(state_dict, i)


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
):
    """Save a state dict to a local directory in safetensors or torch format."""
    logger = get_logger()
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


class WeightCheckpointManager:
    """Utility class to save and cleanup HF-compatible weight checkpoints."""

    def __init__(
        self,
        output_dir: Path,
        config: WeightCheckpointConfig,
        ckpt_config: CheckpointConfig | None,
        async_level: int,
        lora_config: LoRAConfig | None = None,
    ):
        self.weights_dir = get_weights_dir(output_dir)
        self.config = config
        self.ckpt_config = ckpt_config
        self.async_level = async_level
        self.lora_config = lora_config
        self._logger = get_logger()
        self._world = get_world()
        self._is_master = self._world.is_master

    def _get_model_path(self, step: int) -> Path:
        return get_weight_ckpt_model_path(self.weights_dir, step)

    def _get_step_path(self, step: int) -> Path:
        return get_step_path(self.weights_dir, step)

    def _get_adapter_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        """Get adapter weights with clean keys for PEFT compatibility."""
        lora_state = {}

        for key, value in model.state_dict().items():
            param = dict(model.named_parameters()).get(key)
            if param is None or not param.requires_grad:
                continue

            if isinstance(value, DTensor):
                value = value.full_tensor()

            if self._is_master:
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

    def _save_lora_adapters(self, lora_state: dict[str, Tensor], model: nn.Module, step: int):
        """Save LoRA adapters to separate directory."""
        adapter_path = self._get_step_path(step) / "lora_adapters"
        adapter_path.mkdir(parents=True, exist_ok=True)

        torch.save(lora_state, adapter_path / "adapter_model.bin")

        if self.lora_config:
            save_lora_config(self.lora_config, model, adapter_path)  # Pass model

        self._logger.debug(f"Saved LoRA adapters to {adapter_path}")

    def _gather_weights(
        self, model: nn.Module, dtype: torch.dtype = torch.bfloat16, has_lora_layers: bool = False
    ) -> dict[str, Tensor]:
        """Gather distributed weights for weight checkpoint."""
        original_lora_state = None
        if has_lora_layers:
            original_lora_state = merge_lora_weights_inplace(model)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
                warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

                cpu_state = {}
                for key, value in model.state_dict().items():
                    if isinstance(value, DTensor):
                        value = value.to(dtype)
                        # only gather after the downcast to dtype as it will be faster
                        value = value.full_tensor()

                    if self._is_master:
                        key = get_fqns(model, key)
                        assert len(key) == 1
                        key = next(iter(key))
                        # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
                        cpu_state[key] = value.to("cpu", non_blocking=False)

                torch.distributed.barrier()

        finally:
            # Always restore original LoRA state, even if gathering fails
            if original_lora_state is not None:
                restore_lora_weights_inplace(model, original_lora_state)

        # Always clean up the state dict for HF compatibility
        if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in cpu_state.keys()):
            cpu_state = clean_lora_state_dict(cpu_state)

        return cpu_state

    def _save_weights(
        self,
        state_dict: dict[str, Tensor],
        save_dir: Path,
        save_format: Literal["safetensors", "torch"],
        save_sharded: bool,
    ):
        return save_state_dict(state_dict, save_dir, save_format, save_sharded)

    def _save_to_path(
        self,
        state_dict: dict[str, Tensor],
        model,
        tokenizer,
        step: int,
    ):
        """Save weight checkpoint for given step."""
        # Save weight checkpoint temporary dir to avoid race condition
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)

        self._logger.debug(f"Saving weight checkpoint to {step_path}")
        start_time = time.perf_counter()
        # Suppress torch.distributed warnings during checkpoint saving
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

            # Save weights
            self._save_weights(state_dict, step_path, self.config.save_format, self.config.save_sharded)

            # Save model config, generation arguments and tokenizer
            model.config.save_pretrained(step_path)
            if model.generation_config:
                model.generation_config.save_pretrained(step_path)
            tokenizer.save_pretrained(step_path)

        (step_path / "STABLE").touch()
        self._logger.debug(f"Saved weight checkpoint to {step_path} in {time.perf_counter() - start_time:.2f} seconds")

    def create_stable_file(self, step: int):
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        (step_path / "STABLE").touch()

    def save(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        step: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Save a HF-compatible weight-only checkpoint for a given step."""
        has_lora = has_lora_layers(model)

        # Save LoRA adapters separately if configured
        if self.config.save_adapter_separately and has_lora:
            if self._is_master:
                lora_state = self._get_adapter_state_dict(model)
                self._save_lora_adapters(lora_state, model, step)
            torch.distributed.barrier()

        cpu_state = self._gather_weights(model, dtype, has_lora_layers=has_lora)
        if has_tt_moe_layers(cpu_state):
            convert_tt_to_hf_moe(cpu_state)

        if self._is_master:
            if self.config.save_async:
                thread = threading.Thread(
                    target=self._save_to_path,
                    args=(cpu_state, model, tokenizer, step),
                    name=f"weight-checkpoint-save-{step}",
                )
                thread.start()
            else:
                self._save_to_path(cpu_state, model, tokenizer, step)

        return self._get_model_path(step)

    def _maybe_clean(self, step: int):
        """Synchronous helper of `clean`."""
        step = max(step - (self.async_level + 1), 0)  # Consider deleting async_level + 1 steps ago
        candidate_path_to_delete = self._get_step_path(step)
        keep_for_eval = bool(self.config.interval and step % self.config.interval == 0)
        keep_for_ckpt = bool(self.ckpt_config and self.ckpt_config.interval and step % self.ckpt_config.interval == 0)
        self._logger.debug(
            f"Considering deleting weight checkpoint {candidate_path_to_delete} ({keep_for_eval=}, {keep_for_ckpt=})"
        )
        if not (keep_for_eval or keep_for_ckpt):
            self._logger.debug(
                f"Removing past weight checkpoint {candidate_path_to_delete} ({keep_for_eval=}, {keep_for_ckpt=})"
            )
            shutil.rmtree(candidate_path_to_delete, ignore_errors=True)

    def maybe_clean(self, step: int):
        """
        Considers deleting a past weight checkpoint at a given step. There are two reasons not to delete a checkpoint:
        1. The step is an evaluation step (e.g. step % weights.interval == 0)
        2. The step is a checkpoint step or at most async_level steps earlier
        """
        if self.config.save_async:
            thread = threading.Thread(
                target=self._maybe_clean,
                args=(step,),
                name=f"weight-checkpoint-clean-{step}",
            )
            thread.start()
        else:
            self._maybe_clean(step)


def setup_weight_ckpt_manager(
    output_dir: Path,
    weight_ckpt_config: WeightCheckpointConfig | None,
    ckpt_config: CheckpointConfig | None,
    async_level: int,
    lora_config: LoRAConfig | None = None,
) -> WeightCheckpointManager | None:
    if weight_ckpt_config is None:
        return None

    return WeightCheckpointManager(
        output_dir, weight_ckpt_config, ckpt_config, async_level=async_level, lora_config=lora_config
    )
