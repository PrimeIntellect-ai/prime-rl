"""AFMoE-specific attention implementations.

AFMoE attention has several unique features:
- RoPE is only applied to sliding-window (local) attention layers
- Output gating: output = output * sigmoid(gate_proj(hidden_states))
- Configurable sliding window size for local attention layers
- QK normalization is always enabled
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.layers.rms_norm import RMSNorm, RMSNormConfig
from prime_rl.trainer.models.layers.rotary_emb import apply_rotary_pos_emb

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


# =============================================================================
# Compiled Flex Attention (module-level singleton)
# =============================================================================

_COMPILED_FLEX_ATTENTION = None


def _get_compiled_flex_attention():
    """Get the compiled flex_attention function (lazy singleton)."""
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention

            # Use reduce-overhead mode for better handling of variable shapes
            # and dynamic=True to avoid recompilation on shape changes
            _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, mode="default", dynamic=True)
        except ImportError:
            _COMPILED_FLEX_ATTENTION = None
    return _COMPILED_FLEX_ATTENTION


# =============================================================================
# Flex Attention Mask Utilities
# =============================================================================

def _get_causal_mask_mod() -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a causal mask modifier for flex_attention."""

    def causal_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return q_idx >= kv_idx

    return causal_mask


def _get_sliding_window_mask_mod(
    sliding_window: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create a sliding window mask modifier for flex_attention."""

    def sliding_window_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return q_idx - kv_idx <= sliding_window

    return sliding_window_mask


def _and_masks(
    *mask_mods: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Combine multiple mask modifiers with AND logic."""

    def combined_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        result = mask_mods[0](b, h, q_idx, kv_idx)
        for mask_mod in mask_mods[1:]:
            result = result & mask_mod(b, h, q_idx, kv_idx)
        return result

    return combined_mask


@lru_cache(maxsize=8)
def _create_block_mask_cached(
    mask_mod_key: str,
    batch_size: int,
    num_heads: int | None,
    seq_len: int,
    sliding_window: int | None,
    device_str: str,
) -> "BlockMask":
    """Create and cache a BlockMask for flex_attention.

    Args:
        mask_mod_key: Key identifying the mask type ("causal" or "sliding_window")
        batch_size: Batch size
        num_heads: Number of attention heads (None for head-agnostic mask)
        seq_len: Sequence length
        sliding_window: Sliding window size (only used if mask_mod_key == "sliding_window")
        device_str: Device string for placement

    Returns:
        BlockMask for use with flex_attention
    """
    from torch.nn.attention.flex_attention import create_block_mask

    if mask_mod_key == "causal":
        mask_mod = _get_causal_mask_mod()
    elif mask_mod_key == "sliding_window":
        assert sliding_window is not None
        mask_mod = _and_masks(_get_causal_mask_mod(), _get_sliding_window_mask_mod(sliding_window))
    else:
        raise ValueError(f"Unknown mask type: {mask_mod_key}")

    device = torch.device(device_str)
    return create_block_mask(mask_mod, batch_size, num_heads, seq_len, seq_len, device=device)


def create_afmoe_block_masks(
    batch_size: int,
    seq_len: int,
    sliding_window: int,
    device: torch.device,
) -> dict[str, "BlockMask"]:
    """Create BlockMasks for both full and sliding window attention.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        sliding_window: Sliding window size for local attention
        device: Device for mask placement

    Returns:
        Dictionary with "full_attention" and "sliding_attention" BlockMasks
    """
    device_str = str(device)
    
    # Note: BlockMask is batch-size agnostic when B=1, so we always use B=1
    # This allows cache hits even when batch size varies
    # flex_attention will broadcast the mask across the batch dimension
    return {
        "full_attention": _create_block_mask_cached("causal", 1, None, seq_len, None, device_str),
        "sliding_attention": _create_block_mask_cached(
            "sliding_window", 1, None, seq_len, sliding_window, device_str
        ),
    }


@dataclass
class AfmoeAttentionConfig:
    """Configuration for AFMoE attention layers.

    Args:
        hidden_size: Model hidden dimension
        head_dim: Dimension per attention head
        num_attention_heads: Number of query heads
        num_key_value_heads: Number of key/value heads (for GQA)
        rms_norm_eps: Epsilon for RMSNorm
        is_local_attention: Whether this layer uses sliding window attention
        sliding_window: Window size for sliding attention (None for full attention)
        attention_dropout: Dropout probability for attention weights
    """

    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    is_local_attention: bool
    sliding_window: int | None = None
    attention_dropout: float = 0.0


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class AfmoeSDPAAttention(nn.Module):
    """AFMoE attention using PyTorch's scaled_dot_product_attention.

    Features:
    - RoPE only applied when is_local_attention=True
    - Output gating with learned gate projection
    - QK normalization always enabled
    - Sliding window support via attention mask
    """

    def __init__(self, config: AfmoeAttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_local_attention = config.is_local_attention
        self.sliding_window = config.sliding_window if config.is_local_attention else None
        self.attention_dropout = config.attention_dropout

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Output gating
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)

        # QK normalization (always enabled for AFMoE)
        self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        block_mask: "BlockMask | None" = None,  # Ignored for SDPA, for interface compatibility
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            attention_mask: Optional attention mask
            block_mask: Ignored (only used by flex_attention)

        Returns:
            Tuple of (output, None) - None is for attention weights compatibility
        """
        del block_mask  # Unused in SDPA

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project Q, K, V and gate
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        # QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE only to local (sliding window) attention layers
        if self.is_local_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV for GQA
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        # SDPA attention
        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=attention_mask is None,  # Use causal if no explicit mask
            scale=self.scaling,
        )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*input_shape, -1)

        # Apply output gating
        attn_output = attn_output * F.sigmoid(gate_states)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None


class AfmoeFlexAttention(nn.Module):
    """AFMoE attention using PyTorch's flex_attention.

    Features:
    - RoPE only applied when is_local_attention=True
    - Output gating with learned gate projection
    - QK normalization always enabled
    - Native sliding window support via block_mask
    """

    def __init__(self, config: AfmoeAttentionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_local_attention = config.is_local_attention
        self.sliding_window = config.sliding_window if config.is_local_attention else None
        self.attention_dropout = config.attention_dropout

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Output gating
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)

        # QK normalization (always enabled for AFMoE)
        self.q_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))
        self.k_norm = RMSNorm(RMSNormConfig(hidden_size=self.head_dim, eps=config.rms_norm_eps))

        # Use module-level compiled flex_attention singleton
        self._flex_attention = _get_compiled_flex_attention()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        block_mask: "BlockMask | None" = None,
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            attention_mask: Optional dense attention mask (falls back to SDPA if provided)
            block_mask: Optional BlockMask for flex_attention

        Returns:
            Tuple of (output, None) - None is for attention weights compatibility
        """
        if self._flex_attention is None:
            raise RuntimeError("flex_attention requires PyTorch 2.5+ with torch.nn.attention.flex_attention")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project Q, K, V and gate
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        # QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE only to local (sliding window) attention layers
        if self.is_local_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV for GQA
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        # Use flex_attention when block_mask is provided (preferred)
        # Fall back to SDPA only if no block_mask and a dense attention_mask is provided
        if block_mask is not None:
            # Use flex_attention with block_mask
            attn_output = self._flex_attention(
                query_states,
                key_states,
                value_states,
                block_mask=block_mask,
                scale=self.scaling,
            )
        elif attention_mask is not None:
            # Fall back to SDPA with dense mask
            dropout_p = self.attention_dropout if self.training else 0.0
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                scale=self.scaling,
            )
        else:
            # No mask provided, use flex_attention with causal
            attn_output = self._flex_attention(
                query_states,
                key_states,
                value_states,
                block_mask=None,
                scale=self.scaling,
            )

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*input_shape, -1)

        # Apply output gating
        attn_output = attn_output * F.sigmoid(gate_states)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None


# Registry mapping attention implementation names to classes
AFMOE_ATTN_IMPL2CLASS = {
    "sdpa": AfmoeSDPAAttention,
    "flex_attention": AfmoeFlexAttention,
}


def get_afmoe_attention(attn_implementation: str, config: AfmoeAttentionConfig) -> nn.Module:
    """Factory function to create AFMoE attention layer.

    Args:
        attn_implementation: One of "sdpa" or "flex_attention"
        config: AfmoeAttentionConfig instance

    Returns:
        Instantiated attention module

    Raises:
        ValueError: If attn_implementation is not supported
    """
    if attn_implementation not in AFMOE_ATTN_IMPL2CLASS:
        supported = list(AFMOE_ATTN_IMPL2CLASS.keys())
        raise ValueError(
            f"AFMoE attention does not support '{attn_implementation}'. "
            f"Supported implementations: {supported}. "
            f"Note: flash_attention is not supported for AFMoE due to sliding window + RoPE constraints."
        )
    return AFMOE_ATTN_IMPL2CLASS[attn_implementation](config)
