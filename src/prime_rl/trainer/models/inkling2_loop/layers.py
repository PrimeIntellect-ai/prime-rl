"""Building blocks of the Inkling 2 decoder around its attention and MoE.

Rounding follows the serving kernels: norms, attention-residual reads, convolutions and gates compute in FP32 and
round once to the residual's dtype; only GEMMs run in that dtype. Functions that return FP32 say so. This holds in
eager execution. A compiled layer keeps FP32 where Inductor fuses two such operations and skips the rounding between
them, so its outputs differ from eager's at rounding level.

Activations are ``[1, T, ...]``: one packed row of documents. ``positions [T]`` counts tokens inside each document,
which is what keeps a convolution from reaching into the previous document.
"""

import torch
from torch import nn

DEPTH_MIX_EPS = 1e-6
L2_CLIP_EPS = 1e-6


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``x`` scaled to unit RMS over its last dim, in FP32."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


class Inkling2RMSNorm(nn.Module):
    """RMSNorm that takes an FP32 or residual-dtype input and rounds its FP32 result to ``dtype``."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (unweighted_rms_norm(x, self.eps) * self.weight.float()).to(dtype)


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Scale each head vector of ``x [T, heads, d]`` by ``scale [heads, d]``, then shrink it to L2 norm
    ``threshold`` if it is longer. Shorter vectors are unchanged."""
    values = x.float()
    if scale is not None:
        values = values * scale.float()
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return (values * (threshold / (norm + L2_CLIP_EPS)).clamp(max=1.0)).to(x.dtype)


def log_length_scale(positions: torch.Tensor, n_floor: int, alpha: float) -> torch.Tensor:
    """``1 + alpha * log(max((position + 1) / n_floor, 1))`` as FP32 ``[T, 1, 1]``: one up to ``n_floor`` tokens,
    growing past them."""
    return (1.0 + alpha * torch.log(((positions + 1).float() / n_floor).clamp(min=1.0))).view(-1, 1, 1)


def _lagged(x: torch.Tensor, lag: int, positions: torch.Tensor) -> torch.Tensor:
    """``x[:, t - lag]`` where that token is in ``t``'s document, zero elsewhere."""
    if lag == 0:
        return x
    in_document = (positions >= lag).view(1, -1, *([1] * (x.dim() - 2)))
    return torch.roll(x, shifts=lag, dims=1) * in_document


class StaticShortConv(nn.Module):
    """Causal depthwise convolution with an identity residual, scaled by ``(kernel + 1) ** -0.5``."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        # The checkpoint's Conv1d layout: tap `kernel_size - 1` weights the current token.
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))

    def forward(self, x: torch.Tensor, source: torch.Tensor | None, positions: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float().squeeze(1)
        kernel_size = weight.shape[1]
        values = x.float()
        total = values
        for lag in range(kernel_size):
            total = total + _lagged(values, lag, positions) * weight[:, kernel_size - 1 - lag]
        return (total * (kernel_size + 1) ** -0.5).to(x.dtype)


class DynamicShortConv(nn.Module):
    """Causal convolution whose taps are static per-channel weights plus per-token weights projected from
    ``source`` and shared within a head. Identity residual, scaled by ``(2 * kernel + 1) ** -0.5``."""

    def __init__(self, channels: int, source_size: int, num_heads: int, kernel_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Row `lag` weights the token `lag` steps back.
        self.static_weight = nn.Parameter(torch.empty(kernel_size, channels))
        self.weight_proj = nn.Linear(source_size, num_heads * kernel_size, bias=False)

    def forward(self, x: torch.Tensor, source: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        kernel_size = self.static_weight.shape[0]
        token_taps = self.weight_proj(source).float().unflatten(-1, (self.num_heads, 1, kernel_size))
        static_taps = self.static_weight.float().unflatten(-1, (self.num_heads, -1))
        values = x.float().unflatten(-1, (self.num_heads, -1))
        total = values
        for lag in range(kernel_size):
            total = total + _lagged(values, lag, positions) * (static_taps[lag] + token_taps[..., lag])
        return (total * (2 * kernel_size + 1) ** -0.5).flatten(-2).to(x.dtype)


def depth_mix(sources: torch.Tensor, queries: torch.Tensor, source_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Attend over depth: one softmax mix of the unit-RMS ``sources [1, T, n, H]`` per row of ``queries [m, H]``.
    ``source_mask [n]`` marks the sources that exist; the others get no weight.

    Returns FP32 ``[1, T, m, H]``.
    """
    values = unweighted_rms_norm(sources, DEPTH_MIX_EPS)
    scores = torch.matmul(values, queries.float().t()).transpose(-1, -2) * values.shape[-1] ** -0.5
    if source_mask is not None:
        scores = scores.masked_fill(~source_mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), values)


def block_sources(
    block_states: torch.Tensor, block_mask: torch.Tensor, *later: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sources of a depth mix and their mask: the block slots, then the ``later [1, T, H]`` states, in FP32.

    The decoder passes ``block_states [1, T, slots, H]`` with one slot per block of the whole model and
    ``block_mask [slots]`` marking the finished ones, so that every layer sees the same shapes.
    """
    sources = torch.cat([block_states.float(), *(state.float().unsqueeze(2) for state in later)], dim=2)
    return sources, torch.cat([block_mask, block_mask.new_ones(len(later))])


class DepthMix(nn.Module):
    """A sub-layer's input: a learned mix of the finished blocks and the running sum of the current one."""

    def __init__(self, hidden_size: int, num_queries: int = 1) -> None:
        super().__init__()
        self.query = nn.Parameter(
            torch.zeros(hidden_size) if num_queries == 1 else torch.zeros(num_queries, hidden_size)
        )

    def forward(self, sources: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """``sources [1, T, n, H]``, ``source_mask [n]``. Returns FP32 ``[1, T, num_queries, H]``."""
        return depth_mix(sources, self.query.view(-1, self.query.shape[-1]), source_mask)


class MessageGate(nn.Module):
    """Blend a loop message into a sub-layer input: ``(1 - forget) * state + gain * message``, with both gates a
    tanh of a low-rank projection of the normed pair."""

    def __init__(self, hidden_size: int, rank: int, eps: float) -> None:
        super().__init__()
        self.down_proj = nn.Linear(2 * hidden_size, rank, bias=False)
        self.message_proj = nn.Linear(rank, hidden_size, bias=False)
        self.forget_proj = nn.Linear(rank, hidden_size, bias=False)
        self.eps = eps

    def forward(self, state: torch.Tensor, message: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Returns FP32."""
        state, message = state.float(), message.float()
        normed = torch.cat([unweighted_rms_norm(state, self.eps), unweighted_rms_norm(message, self.eps)], dim=-1)
        hidden = self.down_proj(normed.to(dtype)).float()
        forget = torch.tanh(hidden @ self.forget_proj.weight.float().t())
        gain = torch.tanh(hidden @ self.message_proj.weight.float().t())
        return (1.0 - forget) * state + gain * message


class Inkling2FinalNorm(nn.Module):
    """The model's output read: a depth mix over every block and the last running sum, then RMSNorm."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, block_states: torch.Tensor, block_mask: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        sources, source_mask = block_sources(block_states, block_mask, current)
        mixed = depth_mix(sources, self.query.unsqueeze(0), source_mask)[:, :, 0]
        return (unweighted_rms_norm(mixed, self.eps) * self.weight.float()).to(current.dtype)


__all__ = [
    "DepthMix",
    "DynamicShortConv",
    "Inkling2FinalNorm",
    "Inkling2RMSNorm",
    "MessageGate",
    "StaticShortConv",
    "block_sources",
    "depth_mix",
    "l2_clip",
    "log_length_scale",
    "unweighted_rms_norm",
]
