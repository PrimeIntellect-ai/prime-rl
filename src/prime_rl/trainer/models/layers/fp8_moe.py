# Copyright (c) Prime Intellect.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 MoE Expert Implementation using Transformer Engine's GroupedLinear.

This module provides FP8 conversion for MoE expert layers using NVIDIA Transformer Engine's
GroupedLinear with Float8BlockScaling recipe (DeepSeek V3 style blockwise scaling).

The conversion workflow:
1. Initialize model with standard torch format (for weight loading compatibility)
2. Convert GroupedExperts to FP8GroupedExperts before FSDP setup
3. Train with FP8 using TE's autocast with Float8BlockScaling recipe

Blockwise scaling configuration:
- Activations: 1x128 blockwise (x_block_scaling_dim=1)
- Weights: 128x128 blockwise (w_block_scaling_dim=2)
"""

import contextlib
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    pass


def _is_transformer_engine_available() -> bool:
    """Check if Transformer Engine is available with full PyTorch support."""
    try:
        import transformer_engine.pytorch  # noqa: F401
        from transformer_engine.common.recipe import Float8BlockScaling  # noqa: F401
        from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding, GroupedLinear  # noqa: F401

        return True
    except (ImportError, FileNotFoundError, RuntimeError):
        # ImportError: TE not installed
        # FileNotFoundError: TE installed but missing shared object file
        # RuntimeError: TE installation issues
        return False


TE_AVAILABLE = _is_transformer_engine_available()


class FP8GroupedExperts(nn.Module):
    """
    FP8 wrapper for GroupedExperts using Transformer Engine's GroupedLinear.

    This wraps a base GroupedExperts module and replaces the computation with
    FP8 operations using TE's GroupedLinear. State dict hooks ensure compatibility
    with original checkpoint format (w1, w2, w3 stacked tensors).

    The wrapper pattern (similar to LoRA) allows:
    - Loading checkpoints in original format via standard load_state_dict()
    - Saving checkpoints in original format via standard state_dict()
    - Attribute forwarding to base layer for compatibility

    Supports both meta device initialization (for DCP weight loading) and
    regular device initialization (for immediate weight copying).

    Usage:
        # Create base model and load weights
        base_model = GroupedExperts(config).cuda().bfloat16()
        base_model.load_state_dict(checkpoint)

        # Wrap with FP8
        fp8_model = FP8GroupedExperts(base_model)

        # Create optimizer on wrapped model
        optimizer = torch.optim.AdamW(fp8_model.parameters(), lr=1e-4)

        # Train with FP8 autocast context manager (handled by FP8TrainingContextManager)
    """

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> None:
        if not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is required for FP8 MoE training. Install with: pip install transformer-engine"
            )

        from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding, GroupedLinear

        super().__init__()

        num_experts = w1.shape[0]
        dim = w1.shape[2]  # w1 shape: [num_experts, hidden_dim, dim]
        hidden_dim = w1.shape[1]

        self.num_experts = num_experts
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.padding = Fp8Padding(num_gemms=self.num_experts)
        self.unpadding = Fp8Unpadding(num_gemms=self.num_experts)

        # Determine device - use meta if base_layer is on meta, otherwise cuda
        is_meta = w1.device.type == "meta"
        device = "meta" if is_meta else "cuda"

        # Get dtype from base layer weights to ensure consistency with FSDP
        params_dtype = w1.dtype

        # Create TE GroupedLinear modules with same names as original (w1, w2, w3)
        # w1: dim -> hidden_dim (gate projection)
        self.w1 = GroupedLinear(
            num_gemms=num_experts,
            in_features=dim,
            out_features=hidden_dim,
            bias=False,
            device=device,
            params_dtype=params_dtype,
        )

        # w2: hidden_dim -> dim (down projection)
        self.w2 = GroupedLinear(
            num_gemms=num_experts,
            in_features=hidden_dim,
            out_features=dim,
            bias=False,
            device=device,
            params_dtype=params_dtype,
        )

        # w3: dim -> hidden_dim (up projection)
        self.w3 = GroupedLinear(
            num_gemms=num_experts,
            in_features=dim,
            out_features=hidden_dim,
            bias=False,
            device=device,
            params_dtype=params_dtype,
        )

        # Copy weights from base layer to TE modules (only if not on meta device)
        if not is_meta:
            with torch.no_grad():
                for i in range(num_experts):
                    getattr(self.w1, f"weight{i}").copy_(w1.data[i])
                    getattr(self.w2, f"weight{i}").copy_(w2.data[i])
                    getattr(self.w3, f"weight{i}").copy_(w3.data[i])

        # Delete base layer parameters (weights now live in TE modules)
        del w1
        del w2
        del w3

        # Register state dict hooks for checkpoint compatibility
        # Post hook: convert TE format -> original format when saving
        self._register_state_dict_hook(self._post_state_dict_hook)
        # Pre hook: convert original format -> TE format when loading
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    def set_weights(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> None:
        """Set the weights for the TE modules."""
        with torch.no_grad():
            for i in range(self.num_experts):
                getattr(self.w1, f"weight{i}").copy_(w1.data[i])
                getattr(self.w2, f"weight{i}").copy_(w2.data[i])
                getattr(self.w3, f"weight{i}").copy_(w3.data[i])

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        FP8 forward pass using TE GroupedLinear.

        Note: This should be called inside TE's autocast context manager for FP8 execution.

        Args:
            x: (total_tokens, dim) - tokens reordered by expert
            num_tokens_per_expert: (num_experts,) - token count per expert

        Returns:
            (total_tokens, dim) - expert outputs
        """
        original_m_splits = [int(m) for m in num_tokens_per_expert.cpu().tolist()]

        x, m_splits = self.padding(x, original_m_splits)

        # w1 projection (gate)
        w1_out = self.w1(x, m_splits=m_splits)

        # w3 projection (up)
        w3_out = self.w3(x, m_splits=m_splits)

        # SwiGLU activation
        h = F.silu(w1_out) * w3_out

        # w2 projection (down)
        out = self.w2(h, m_splits=m_splits)

        out = self.unpadding(out, m_splits=original_m_splits)
        return out

    def init_weights(self, init_std: float):
        """Initialize weights for TE modules."""
        for i in range(self.num_experts):
            nn.init.trunc_normal_(getattr(self.w1, f"weight{i}"), mean=0.0, std=0.02)
            nn.init.trunc_normal_(getattr(self.w2, f"weight{i}"), mean=0.0, std=init_std)
            nn.init.trunc_normal_(getattr(self.w3, f"weight{i}"), mean=0.0, std=init_std)

    @staticmethod
    def _post_state_dict_hook(
        module: "FP8GroupedExperts",
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *args,
    ) -> dict[str, torch.Tensor]:
        """
        Convert TE format to original format when saving state dict.

        This allows checkpoints to be saved in the original format (w1, w2, w3)
        so they can be loaded by non-FP8 models.

        TE format: w1.weight0, w1.weight1, ... (individual expert weights)
        Original format: w1 with shape (num_experts, out_features, in_features)
        """
        num_experts = module.num_experts

        for weight_name in ["w1", "w2", "w3"]:
            # Collect all expert weights for this projection
            expert_weights = []
            keys_to_remove = []

            for i in range(num_experts):
                te_key = f"{prefix}{weight_name}.weight{i}"
                if te_key in state_dict:
                    expert_weights.append(state_dict[te_key])
                    keys_to_remove.append(te_key)

            # Stack into original format and add to state dict
            if expert_weights:
                orig_key = f"{prefix}{weight_name}"
                state_dict[orig_key] = torch.stack(expert_weights, dim=0)

                # Remove individual expert weight keys
                for key in keys_to_remove:
                    del state_dict[key]

            # Also remove _extra_state keys from TE modules
            extra_state_key = f"{prefix}{weight_name}._extra_state"
            if extra_state_key in state_dict:
                del state_dict[extra_state_key]

        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: "FP8GroupedExperts",
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *args,
    ) -> None:
        """
        Convert original format to TE format when loading state dict.

        This allows loading checkpoints in original format (w1, w2, w3)
        into the FP8 model.

        Original format: w1 with shape (num_experts, out_features, in_features)
        TE format: w1.weight0, w1.weight1, ... (individual expert weights)
        """
        num_experts = module.num_experts
        logger = get_logger()
        logger.warning(f"Loading state dict with prefix: {prefix}")

        for weight_name in ["w1", "w2", "w3"]:
            orig_key = f"{prefix}{weight_name}"

            if orig_key in state_dict:
                logger.warning(f"Found {orig_key} in state dict")
                # Get stacked weight tensor
                stacked_weight = state_dict[orig_key]  # (num_experts, out_features, in_features)

                # Split into individual expert weights
                for i in range(num_experts):
                    te_key = f"{prefix}{weight_name}.weight{i}"
                    state_dict[te_key] = stacked_weight[i]

                # Remove original key
                del state_dict[orig_key]


def _is_block_scaling_supported() -> bool:
    """Check if Float8BlockScaling is supported (requires CUDA >= 12.9 and compute capability >= 9.0)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False

        # Check compute capability (need >= 9.0 for Hopper+)
        cc_major, _ = torch.cuda.get_device_capability()
        if cc_major < 9:
            return False

        # Check CUDA version (need >= 12.9 for block scaling)
        cuda_version = torch.version.cuda
        if cuda_version is None:
            return False

        major, minor = map(int, cuda_version.split(".")[:2])
        cuda_version_num = major * 10 + minor
        if cuda_version_num < 129:  # 12.9
            return False

        return True
    except Exception:
        return False


BLOCK_SCALING_SUPPORTED = _is_block_scaling_supported()


class FP8TrainingContextManager:
    """
    Context manager for FP8 training using Transformer Engine.

    This wraps the forward pass in TE's autocast with an appropriate FP8 recipe.
    - For CUDA >= 12.9: Uses Float8BlockScaling (DeepSeek V3 style)
    - For CUDA < 12.9: Falls back to DelayedScaling

    Usage:
        fp8_ctx = FP8TrainingContextManager()

        for batch in dataloader:
            with fp8_ctx():
                loss = model(batch)
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        enabled: bool = True,
        x_block_scaling_dim: int = 1,  # 1x128 for activations (only for block scaling)
        w_block_scaling_dim: int = 2,  # 128x128 for weights (only for block scaling)
        grad_block_scaling_dim: int = 1,  # 1x128 for gradients (only for block scaling)
    ):
        if not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is required for FP8 training. Install with: pip install transformer-engine"
            )

        from transformer_engine.pytorch import autocast

        self.enabled = enabled
        self.autocast = autocast

        from transformer_engine.common.recipe import Float8BlockScaling

        self.recipe = Float8BlockScaling(
            x_block_scaling_dim=x_block_scaling_dim,
            w_block_scaling_dim=w_block_scaling_dim,
            grad_block_scaling_dim=grad_block_scaling_dim,
        )
        self.recipe_type = "Float8BlockScaling"

    def __call__(self):
        """Return the autocast context manager."""
        return self.autocast(enabled=self.enabled, recipe=self.recipe)


def convert_grouped_experts_to_fp8(w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> FP8GroupedExperts:
    """
    Convert a GroupedExperts module to FP8GroupedExperts.

    Args:
        experts: The GroupedExperts module to convert

    Returns:
        FP8GroupedExperts wrapper with the same weights
    """
    if not TE_AVAILABLE:
        raise ImportError(
            "Transformer Engine is required for FP8 MoE conversion. Install with: pip install transformer-engine"
        )

    return FP8GroupedExperts(w1, w2, w3)


def convert_model_moe_to_fp8(model: nn.Module) -> None:
    """
    Convert all MoE expert layers in a model to FP8.

    This function traverses the model and replaces all GroupedExperts modules
    with FP8GroupedExperts wrappers. The conversion happens in-place.

    Args:
        model: The model to convert. Must have transformer blocks with MoE layers.
    """
    from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE

    logger = get_logger()

    if not TE_AVAILABLE:
        raise ImportError(
            "Transformer Engine is required for FP8 MoE conversion. Install with: pip install transformer-engine"
        )

    num_converted = 0
    for transformer_block in model.model.layers:
        if isinstance(transformer_block.mlp, MoE):
            if isinstance(transformer_block.mlp.experts, GroupedExperts):
                # Convert to FP8
                fp8_experts = FP8GroupedExperts(
                    transformer_block.mlp.experts.w1, transformer_block.mlp.experts.w2, transformer_block.mlp.experts.w3
                )
                transformer_block.mlp.__delattr__("experts")
                transformer_block.mlp.register_module("experts", fp8_experts)
                num_converted += 1

    if num_converted > 0:
        logger.info(f"Converted {num_converted} MoE expert layers to FP8 (TE GroupedLinear)")
    else:
        logger.warning("No MoE expert layers found to convert to FP8")


def is_fp8_moe_model(model: nn.Module) -> bool:
    """Check if the model has FP8 MoE layers."""
    from prime_rl.trainer.models.layers.moe import MoE

    for transformer_block in model.model.layers:
        if isinstance(transformer_block.mlp, MoE):
            if isinstance(transformer_block.mlp.experts, FP8GroupedExperts):
                return True
    return False


def get_fp8_context_manager(enabled: bool = True) -> FP8TrainingContextManager | None:
    """
    Get an FP8 training context manager if enabled and available.

    Args:
        enabled: Whether FP8 training is enabled

    Returns:
        FP8TrainingContextManager if enabled and TE is available, None otherwise
    """
    if not enabled:
        return contextlib.nullcontext

    if not TE_AVAILABLE:
        return contextlib.nullcontext

    return FP8TrainingContextManager(enabled=True)
