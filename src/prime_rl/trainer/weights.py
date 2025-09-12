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
from prime_rl.trainer.lora import LoRALinear
from prime_rl.trainer.rl.config import WeightCheckpointConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_step_path, get_weight_ckpt_model_path, get_weights_dir
from .quant import quantize_to_q5_0_per_channel, should_quantize_tensor


def _has_tt_moe_layers(state_dict: dict[str, Tensor]) -> bool:
    return any("mlp.router.gate" in i for i in state_dict.keys())


def _has_lora_layers(model: nn.Module) -> bool:
    """Check if model has LoRA layers."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            return True
    return False


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


def _clean_lora_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
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


def _merge_lora_weights_inplace(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """
    Merge LoRA weights into base layers in-place and return original LoRA state for restoration.

    Returns:
        Dictionary mapping module names to their original LoRA state
    """
    original_lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            original_lora_state[name] = {
                "lora_A": module.lora_A.data.clone(),
                "lora_B": module.lora_B.data.clone(),
            }

            delta_weight = (module.lora_B @ module.lora_A) * module.scaling
            module.base_layer.weight.data.add_(delta_weight)

            module.lora_A.data.zero_()
            module.lora_B.data.zero_()

    return original_lora_state


def _restore_lora_weights_inplace(model: nn.Module, original_lora_state: dict[str, dict[str, torch.Tensor]]) -> None:
    """
    Restore original LoRA weights and subtract merged weights from base layers.

    Args:
        model: Model with merged LoRA weights
        original_lora_state: Original LoRA state from _merge_lora_weights_inplace
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in original_lora_state:
            module.lora_A.data.copy_(original_lora_state[name]["lora_A"])
            module.lora_B.data.copy_(original_lora_state[name]["lora_B"])

            delta_weight = (module.lora_B @ module.lora_A) * module.scaling
            module.base_layer.weight.data.sub_(delta_weight)


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

    def _gather_weights(
        self, model: nn.Module, dtype: torch.dtype = torch.bfloat16, has_lora_layers: bool = False, quantize_q5: bool = True
    ) -> dict[str, Tensor]:
        """Gather distributed weights for weight checkpoint with non-blocking transfers and optional Q5_0 per-channel quantization."""
        original_lora_state = None
        if has_lora_layers:
            original_lora_state = _merge_lora_weights_inplace(model)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
                warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

                cpu_state = {}
                quantized_count = 0
                total_count = 0
                total_size_before = 0
                total_size_after = 0
                has_cuda_transfers = False
                
                for key, value in model.state_dict().items():
                    if isinstance(value, DTensor):
                        # Handle DTensor from FSDP2 - gather first, then quantize
                        if quantize_q5 and should_quantize_tensor(key, value):
                            # Convert to dtype and gather
                            full_tensor = value.to(dtype).full_tensor()
                            
                            if self._is_master:
                                try:
                                    # Quantize the full gathered tensor
                                    packed_value, scales, original_shape = quantize_to_q5_0_per_channel(full_tensor)
                                    
                                    fqn_key = get_fqns(model, key)
                                    assert len(fqn_key) == 1, f"Expected single FQN for {key}, got {len(fqn_key)}"
                                    fqn_key = next(iter(fqn_key))
                                    
                                    # Store packed weights, scales, and shape with non-blocking transfers
                                    if packed_value.is_cuda:
                                        has_cuda_transfers = True
                                    if scales.is_cuda:
                                        has_cuda_transfers = True
                                        
                                    cpu_state[fqn_key] = packed_value.to("cpu", non_blocking=True)
                                    cpu_state[f"{fqn_key}_scales"] = scales.to("cpu", non_blocking=True)
                                    cpu_state[f"{fqn_key}_shape"] = torch.tensor(original_shape, dtype=torch.int64)
                                    
                                    # Log compression statistics
                                    original_size = full_tensor.numel() * 2  # bfloat16 = 2 bytes
                                    compressed_size = packed_value.numel() * 1 + scales.numel() * 4 + 8 * len(original_shape)  # uint8 + float32 scales + int64 shape
                                    total_size_before += original_size
                                    total_size_after += compressed_size
                                    
                                    quantized_count += 1
                                    
                                except Exception as e:
                                    self._logger.error(f"Failed to quantize tensor {key}: {e}")
                                    raise
                            else:
                                # Non-master ranks still need to participate in gather but don't process result
                                pass
                        else:
                            # Regular processing without quantization
                            full_tensor = value.to(dtype).full_tensor()
                            
                            if self._is_master:
                                fqn_key = get_fqns(model, key)
                                assert len(fqn_key) == 1, f"Expected single FQN for {key}, got {len(fqn_key)}"
                                fqn_key = next(iter(fqn_key))
                                
                                if full_tensor.is_cuda:
                                    has_cuda_transfers = True
                                cpu_state[fqn_key] = full_tensor.to("cpu", non_blocking=True)
                                total_size_before += full_tensor.numel() * 2
                                total_size_after += full_tensor.numel() * 2
                        
                        total_count += 1
                    else:
                        # Handle regular tensors
                        if self._is_master:
                            fqn_key = get_fqns(model, key)
                            assert len(fqn_key) == 1, f"Expected single FQN for {key}, got {len(fqn_key)}"
                            fqn_key = next(iter(fqn_key))
                            
                            if quantize_q5 and should_quantize_tensor(key, value):
                                try:
                                    packed_value, scales, original_shape = quantize_to_q5_0_per_channel(value)
                                    
                                    if packed_value.is_cuda:
                                        has_cuda_transfers = True
                                    if scales.is_cuda:
                                        has_cuda_transfers = True
                                        
                                    cpu_state[fqn_key] = packed_value.to("cpu", non_blocking=True)
                                    cpu_state[f"{fqn_key}_scales"] = scales.to("cpu", non_blocking=True)
                                    cpu_state[f"{fqn_key}_shape"] = torch.tensor(original_shape, dtype=torch.int64)
                                    
                                    # Log compression statistics
                                    original_size = value.numel() * 2  # bfloat16 = 2 bytes
                                    compressed_size = packed_value.numel() * 1 + scales.numel() * 4 + 8 * len(original_shape)  # uint8 + float32 scales + int64 shape
                                    total_size_before += original_size
                                    total_size_after += compressed_size
                                    
                                    quantized_count += 1
                                    
                                except Exception as e:
                                    self._logger.error(f"Failed to quantize tensor {key}: {e}")
                                    raise
                            else:
                                if value.is_cuda:
                                    has_cuda_transfers = True
                                cpu_state[fqn_key] = value.to("cpu", non_blocking=True)
                                total_size_before += value.numel() * 2
                                total_size_after += value.numel() * 2
                            
                            total_count += 1

                # Only synchronize if we actually had CUDA transfers
                if self._is_master and has_cuda_transfers:
                    torch.cuda.synchronize()

                torch.distributed.barrier()
                
                if self._is_master and quantize_q5 and quantized_count > 0:
                    compression_ratio = total_size_before / total_size_after if total_size_after > 0 else 1.0
                    size_mb_before = total_size_before / (1024 * 1024)
                    size_mb_after = total_size_after / (1024 * 1024)
                    self._logger.info(
                        f"Q5_0 quantization: {quantized_count}/{total_count} tensors, "
                        f"{size_mb_before:.1f}MB -> {size_mb_after:.1f}MB "
                        f"(compression ratio: {compression_ratio:.2f}x)"
                    )

        finally:
            # Always restore original LoRA state, even if gathering fails
            if original_lora_state is not None:
                _restore_lora_weights_inplace(model, original_lora_state)

        # Always clean up the state dict for HF compatibility
        if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in cpu_state.keys()):
            cpu_state = _clean_lora_state_dict(cpu_state)

        return cpu_state

    def _save_to_path(self, cpu_state: dict[str, Tensor], model: nn.Module, tokenizer: PreTrainedTokenizer, step: int):
        """Save weight checkpoint for given step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)

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

    def save(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        step: int,
        dtype: torch.dtype = torch.bfloat16,
        quantize_q5: bool = True,
    ):
        """Save a HF-compatible weight-only checkpoint for a given step with optional Q5_0 per-channel quantization."""
        has_lora = _has_lora_layers(model)

        cpu_state = self._gather_weights(model, dtype, has_lora_layers=has_lora, quantize_q5=quantize_q5)
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
