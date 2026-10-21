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


def fp8_round_trip(states: torch.Tensor, stream_count: int) -> torch.Tensor:
    """The expanded residual through FP8 e4m3, each (token, stream) row with its own absmax scale; the backward passes
    the gradient straight through. Written as x + (q(x) - x).detach() rather than a custom autograd Function, which
    the per-layer compiled training graphs dropped."""
    streams = states.float().unflatten(-1, (stream_count, -1))
    scale = streams.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / torch.finfo(torch.float8_e4m3fn).max
    quantized = (streams.detach() / scale).to(torch.float8_e4m3fn).float() * scale
    return (streams + (quantized - streams.detach())).flatten(-2).to(states.dtype)


class HyperConnection(nn.Module):
    """Learned mixing and residual injection across expanded residual streams.

    The residual holds ``stream_count`` streams of width ``stream_width``. A read returns ``read_count`` block inputs of
    width ``hidden_size``: each is ``hidden_size / stream_width`` chunks, every chunk a gated mixture of all streams. The
    reads' gates come from one shared low-rank bottleneck, or with ``separate_down`` from one bottleneck per read.
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
        separate_down: bool = False,
        write_count: int = 1,
        conv_kernel_size: int = 0,
        write_mode: str | None = None,
        write_output_zero_init: bool = True,
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
        self.separate_down = separate_down
        if separate_down:
            self.input_mix_weight_down = nn.Linear(expanded_size, read_count * low_rank, bias=False)
            self.input_mix_weight_up = nn.ModuleList(
                nn.Linear(low_rank, self.chunk_count * expanded_size, bias=False) for _ in range(read_count)
            )
        else:
            self.input_mix_weight_down = nn.Linear(expanded_size, low_rank, bias=False)
            self.input_mix_weight_up = nn.Linear(low_rank, read_count * self.chunk_count * expanded_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(expanded_size, write_count * stream_count * self.chunk_count, bias=False)
            if with_residual_injection and write_mode != "output_only"
            else None
        )
        self.read_conv_weight = (
            nn.Parameter(torch.empty(3 * hidden_size, conv_kernel_size)) if conv_kernel_size else None
        )
        # Write variants (sequential blocks, full-width streams): `norm` RMS-normalizes the block output before the write;
        # `output` adds W_o g(y) to the write logits, g an RMS norm of the block output y used only for the logits (y is
        # written as is; with `write_output_zero_init` W_o starts at 0, so the write starts as GR); `output_only` takes
        # the logits from W_o g(y) alone (no dependence on the streams); `channel` replaces the scalar per stream by a
        # gate per stream and channel from the read's bottleneck.
        self.write_mode = write_mode
        self.write_output_zero_init = write_output_zero_init
        logit_norm = write_mode in ("norm", "output", "output_only")
        self.write_norm = ExpandedRMSNorm(hidden_size, 1, norm_eps) if logit_norm else None
        output_term = write_mode in ("output", "output_only")
        self.write_output_weight = nn.Linear(hidden_size, stream_count, bias=False) if output_term else None
        self.write_gate_weight_up = nn.Linear(low_rank, expanded_size, bias=False) if write_mode == "channel" else None

    def read(self, expanded_states: torch.Tensor) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """The ``read_count`` block inputs, and the state ``combine`` needs."""
        normalized_states = self.hc_norm(expanded_states)
        bottleneck = F.silu(self.input_mix_weight_down(normalized_states) / self.stream_count)
        if self.separate_down:
            up_projections = zip(self.input_mix_weight_up, bottleneck.chunk(self.read_count, dim=-1))
            mixing_weights = torch.sigmoid(torch.cat([up(rank) for up, rank in up_projections], dim=-1))
        else:
            mixing_weights = torch.sigmoid(self.input_mix_weight_up(bottleneck))
        normalized_streams = normalized_states.unflatten(-1, (self.stream_count, self.stream_width))
        chunks = [
            (gate.unflatten(-1, (self.stream_count, self.stream_width)) * normalized_streams).mean(dim=-2)
            for gate in mixing_weights.chunk(self.read_count * self.chunk_count, dim=-1)
        ]
        block_inputs = [
            torch.cat(chunks[start : start + self.chunk_count], dim=-1).to(expanded_states.dtype)
            for start in range(0, len(chunks), self.chunk_count)
        ]
        if self.write_mode == "channel":
            return block_inputs, (expanded_states, normalized_states, bottleneck)
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
        residual_state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        expanded_states, normalized_states = residual_state[:2]
        residual_streams = expanded_states.unflatten(-1, (self.stream_count, self.stream_width))
        if self.write_mode == "norm":
            block_output = self.write_norm(block_output)
        if self.write_mode == "channel":
            gates = 2.0 * torch.sigmoid(self.write_gate_weight_up(residual_state[2]))
            gates = gates.unflatten(-1, (self.stream_count, self.stream_width))
            return (residual_streams + block_output.unsqueeze(-2) * gates).flatten(-2)
        if self.write_mode == "output_only":
            logits = self.write_output_weight(self.write_norm(block_output))
        else:
            logits = self.block_inject_weight(normalized_states)
        if self.write_mode == "output":
            logits = logits + self.write_output_weight(self.write_norm(block_output))
        injection_weights = 2.0 * torch.sigmoid(logits / self.stream_count)
        injection_weights = injection_weights.unflatten(-1, (self.write_count, self.stream_count, self.chunk_count))
        block_outputs = block_output if isinstance(block_output, tuple) else (block_output,)
        for output_index, output in enumerate(block_outputs):
            for chunk_index, chunk in enumerate(output.chunk(self.chunk_count, dim=-1)):
                weights = injection_weights[..., output_index, :, chunk_index]
                residual_streams = residual_streams + chunk.unsqueeze(-2) * weights.unsqueeze(-1)
        return residual_streams.flatten(-2)


__all__ = ["ExpandedRMSNorm", "HyperConnection", "fp8_round_trip"]
