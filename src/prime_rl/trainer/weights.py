import shutil
import threading
import time
import warnings
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.config import CheckpointConfig
from prime_rl.trainer.rl.config import WeightCheckpointConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_step_path, get_weight_ckpt_model_path, get_weights_dir


def _has_tt_moe_layers(state_dict: dict[str, Tensor]) -> bool:
    return any("mlp.router.gate" in i for i in state_dict.keys())


def _get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def _convert_tt_moe_to_hf_(state_dict: dict[str, Tensor]):
    num_layers = _get_max_layer_num(state_dict)
    for i in range(num_layers):
        if not f"model.layers.{i}.mlp.router.gate.weight" in state_dict:
            continue  # Not a TT-MoE layer

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
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_expert.w2"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_expert.w3"
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


class WeightCheckpointManager:
    """Utility class to save and cleanup HF-compatible weight checkpoints."""

    def __init__(
        self, output_dir: Path, config: WeightCheckpointConfig, ckpt_config: CheckpointConfig | None, async_level: int
    ):
        self.weights_dir = get_weights_dir(output_dir)
        self.config = config
        self.ckpt_config = ckpt_config
        self.async_level = async_level
        self._logger = get_logger()
        self._world = get_world()
        self._is_master = self._world.is_master

    def _get_model_path(self, step: int) -> Path:
        return get_weight_ckpt_model_path(self.weights_dir, step)

    def _get_step_path(self, step: int) -> Path:
        return get_step_path(self.weights_dir, step)

    def _gather_weights(self, model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> dict[str, Tensor]:
        """Gather distributed weights for weight checkpoint."""
        start_time = time.time()
        self._logger.debug("Gathering sharded weights")

        # Suppress torch.distributed warnings during checkpoint saving
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

        self._logger.debug(f"Gathered sharded weights in {time.time() - start_time:.2f} seconds")

        return cpu_state

    def _save_to_path(self, cpu_state: dict[str, Tensor], model: nn.Module, tokenizer: PreTrainedTokenizer, step: int):
        """Save weight checkpoint for given step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)

        self._logger.debug(f"Saving weight checkpoint to {step_path}")
        start_time = time.time()

        # Suppress torch.distributed warnings during checkpoint saving
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

            # Save model weights to temporary file to avoid race condition
            model_path = self._get_model_path(step)
            tmp_model_path = model_path.with_suffix(".tmp")
            torch.save(cpu_state, tmp_model_path)
            # Rename temporary file to indicate checkpoint is complete
            tmp_model_path.rename(model_path)

            # Save model config, generation arguments and tokenizer
            model.config.save_pretrained(step_path)
            if model.generation_config:
                model.generation_config.save_pretrained(step_path)
            tokenizer.save_pretrained(step_path)

        self._logger.debug(f"Saved weight checkpoint to {step_path} in {time.time() - start_time:.2f} seconds")

    def save(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        step: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Save a HF-compatible weight-only checkpoint for a given step."""
        cpu_state = self._gather_weights(model, dtype)
        if _has_tt_moe_layers(cpu_state):
            _convert_tt_moe_to_hf_(cpu_state)

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
        keep_for_eval = self.config.interval and step % self.config.interval == 0
        # For checkpointing step x, we need all weight checkpoints in [x-async_level, x] (for logprob model)
        # To get [n-k, n] with interval n and buffer k over all natural numbers x, we use the condition (n - (x % n)) % n <= k
        keep_for_ckpt = (
            self.ckpt_config
            and self.ckpt_config.interval
            and (self.ckpt_config.interval - (step % self.ckpt_config.interval)) % self.ckpt_config.interval
            <= self.async_level
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
    weight_ckpt_config: WeightCheckpointConfig,
    ckpt_config: CheckpointConfig | None,
    async_level: int,
) -> WeightCheckpointManager:
    return WeightCheckpointManager(output_dir, weight_ckpt_config, ckpt_config, async_level=async_level)
