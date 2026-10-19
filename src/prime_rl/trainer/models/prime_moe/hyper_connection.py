import torch
import torch.nn.functional as F
from fla.modules.conv import causal_conv1d
from torch import nn


class ExpandedRMSNorm(nn.Module):
    """RMS-normalize each residual stream with an independent affine weight."""

    def __init__(self, hidden_size: int, stream_count: int, eps: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.stream_count = stream_count
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(stream_count * hidden_size))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        streams = hidden_states.float().unflatten(-1, (self.stream_count, self.hidden_size))
        variance = streams.square().mean(dim=-1, keepdim=True)
        normalized = streams * torch.rsqrt(variance + self.eps)
        return (normalized.flatten(-2) * (1.0 + self.weight.float())).to(input_dtype)


class HyperConnection(nn.Module):
    """Learned mixing and residual injection across expanded residual streams.

    The residual holds ``stream_count`` streams of width ``stream_width``. A read returns ``read_count`` block inputs of
    width ``hidden_size``: each is ``hidden_size / stream_width`` chunks, every chunk a gated mixture of all streams.
    The write adds each chunk of each of ``write_count`` block outputs to every stream with one scalar per (output,
    stream, chunk). With ``conv_kernel_size``, the attention's reads are followed by three causal depthwise
    convolutions over the sequence (one per query, key and value input; residual form), which then make up the
    attention's inputs."""

    def __init__(
        self,
        *,
        hidden_size: int,
        stream_count: int,
        low_rank: int,
        norm_eps: float,
        with_residual_injection: bool = True,
        stream_width: int | None = None,
        read_count: int = 1,
        write_count: int = 1,
        conv_kernel_size: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.stream_count = stream_count
        self.stream_width = stream_width or hidden_size
        self.chunk_count = hidden_size // self.stream_width
        self.read_count = read_count
        self.write_count = write_count
        expanded_size = stream_count * self.stream_width

        self.hc_norm = ExpandedRMSNorm(self.stream_width, stream_count, norm_eps)
        self.input_mix_weight_down = nn.Linear(expanded_size, low_rank, bias=False)
        self.input_mix_weight_up = nn.Linear(low_rank, read_count * self.chunk_count * expanded_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(expanded_size, write_count * stream_count * self.chunk_count, bias=False)
            if with_residual_injection
            else None
        )
        self.read_conv_weight = (
            nn.Parameter(torch.empty(3 * hidden_size, conv_kernel_size)) if conv_kernel_size else None
        )

    def read(self, expanded_states: torch.Tensor) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """The ``read_count`` block inputs, and the state ``combine`` needs."""
        normalized_states = self.hc_norm(expanded_states)
        mixing_weights = F.silu(self.input_mix_weight_down(normalized_states) / self.stream_count)
        mixing_weights = torch.sigmoid(self.input_mix_weight_up(mixing_weights))
        normalized_streams = normalized_states.unflatten(-1, (self.stream_count, self.stream_width))
        chunks = [
            (gate.unflatten(-1, (self.stream_count, self.stream_width)) * normalized_streams).mean(dim=-2)
            for gate in mixing_weights.chunk(self.read_count * self.chunk_count, dim=-1)
        ]
        block_inputs = [
            torch.cat(chunks[start : start + self.chunk_count], dim=-1).to(expanded_states.dtype)
            for start in range(0, len(chunks), self.chunk_count)
        ]
        return block_inputs, (expanded_states, normalized_states)

    def attention_inputs(
        self, reads: list[torch.Tensor], cu_seqlens: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """The attention's input(s) from its one or three reads: the read convolutions applied, if any."""
        if self.read_conv_weight is not None:
            inputs = torch.cat(reads * (3 // len(reads)), dim=-1)
            mixed, _ = causal_conv1d(x=inputs, weight=self.read_conv_weight, cu_seqlens=cu_seqlens)
            reads = (inputs + mixed).split(self.hidden_size, dim=-1)
        return reads[0] if len(reads) == 1 else tuple(reads)

    def mix(
        self, expanded_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]:
        reads, residual_state = self.read(expanded_states)
        return self.attention_inputs(reads, cu_seqlens), residual_state

    def forward(self, expanded_states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.mix(expanded_states)

    def combine(
        self,
        block_output: torch.Tensor | tuple[torch.Tensor, ...],
        residual_state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        expanded_states, normalized_states = residual_state
        injection_weights = 2.0 * torch.sigmoid(self.block_inject_weight(normalized_states) / self.stream_count)
        injection_weights = injection_weights.unflatten(-1, (self.write_count, self.stream_count, self.chunk_count))
        residual_streams = expanded_states.unflatten(-1, (self.stream_count, self.stream_width))
        block_outputs = block_output if isinstance(block_output, tuple) else (block_output,)
        for output_index, output in enumerate(block_outputs):
            for chunk_index, chunk in enumerate(output.chunk(self.chunk_count, dim=-1)):
                weights = injection_weights[..., output_index, :, chunk_index]
                residual_streams = residual_streams + chunk.unsqueeze(-2) * weights.unsqueeze(-1)
        return residual_streams.flatten(-2)


__all__ = ["ExpandedRMSNorm", "HyperConnection"]
