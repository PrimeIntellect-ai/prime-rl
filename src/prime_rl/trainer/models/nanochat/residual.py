"""Residual connections of the nanochat block, selected with ``config.residual["type"]``.

A connection owns how the residual stream is read and written. ``enter`` applies the per-layer scalars, ``read`` returns
the sublayer input and any context ``write`` needs, ``write`` folds the sublayer output back. The state carries
``streams`` residual streams of ``hidden_size`` channels, as a tensor of shape ``[..., streams * hidden_size]``.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor


def local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def split_streams(state: torch.Tensor, streams: int, hidden_size: int) -> torch.Tensor:
    return state.unflatten(-1, (streams, hidden_size))


class Connection(nn.Module):
    streams: int = 1
    collapses_streams: bool = False

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def initial_state(self, x0: torch.Tensor) -> torch.Tensor:
        return x0.repeat(1, 1, self.streams) if self.streams > 1 else x0

    def enter(
        self, state: torch.Tensor, x0: torch.Tensor, resid_lambda: torch.Tensor, x0_lambda: torch.Tensor
    ) -> torch.Tensor:
        return resid_lambda.to(state.dtype) * state + x0_lambda.to(state.dtype) * x0.repeat(1, 1, self.streams)

    def reset_parameters(self, generator: torch.Generator) -> None:
        pass

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return rms_norm(state), None

    def write(
        self, state: torch.Tensor, output: torch.Tensor, context: torch.Tensor | None, hidden: torch.Tensor | None
    ) -> torch.Tensor:
        return state + output

    def extra_flops_per_token(self) -> int:
        """FLOPs per token beyond the 6-per-parameter count of the connection's parameters."""
        return 0


class PlainConnection(Connection):
    pass


class GatedConnection(Connection):
    """Gated residual of Qwen3.8-Next (``HyperConnection``). ``streams`` residual streams are carried across blocks; a
    sublayer reads a channel-wise gated mean of the RMS-normalized streams, and its output is added to every stream,
    scaled by a stream-wise gate in (0, 2). Without injection this is the mixer that collapses the streams."""

    def __init__(
        self, hidden_size: int, streams: int, low_rank: int, injection: bool = True, read_scale: float = 1.0
    ) -> None:
        super().__init__(hidden_size)
        self.streams = streams
        self.read_scale = read_scale
        expanded = streams * hidden_size
        self.norm_weight = nn.Parameter(torch.zeros(expanded))
        self.down = nn.Linear(expanded, low_rank, bias=False)
        self.up = nn.Linear(low_rank, expanded, bias=False)
        self.inject = nn.Linear(expanded, streams, bias=False) if injection else None
        self.collapses_streams = not injection

    def reset_parameters(self, generator: torch.Generator) -> None:
        local(self.norm_weight).zero_()
        for module in (self.down, self.up, self.inject):
            if module is not None:
                local(module.weight).normal_(0.0, 0.02, generator=generator)

    def _normalized(self, state: torch.Tensor) -> torch.Tensor:
        streams = split_streams(state.float(), self.streams, self.hidden_size)
        normalized = streams * torch.rsqrt(streams.square().mean(dim=-1, keepdim=True) + 1e-5)
        return (normalized.flatten(-2) * (1.0 + self.norm_weight.float())).to(state.dtype)

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self._normalized(state)
        gates = torch.sigmoid(self.up(F.silu(self.down(normalized) / self.streams)))
        gates = split_streams(gates, self.streams, self.hidden_size)
        block_input = (gates * split_streams(normalized, self.streams, self.hidden_size)).mean(dim=-2)
        injection = 2.0 * torch.sigmoid(self.inject(normalized) / self.streams) if self.inject is not None else None
        return block_input * self.read_scale, injection

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        streams = split_streams(state, self.streams, self.hidden_size)
        return (streams + output.unsqueeze(-2) * context.unsqueeze(-1)).flatten(-2)

    def extra_flops_per_token(self) -> int:
        # The per-stream RMS norm and the channel-wise gates.
        return 12 * self.streams * self.hidden_size


class OrthogonalConnection(Connection):
    """Orthogonal residual update (Oh and Kim, 2025): only the component of the output orthogonal to the stream is
    added, per token. ``sublayers`` restricts it to "attn", "mlp" or "both"; the other sublayer stays plain."""

    def __init__(self, hidden_size: int, sublayer: int, sublayers: str) -> None:
        super().__init__(hidden_size)
        self.active = sublayers == "both" or (sublayers == "attn") == (sublayer == 0)

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        if not self.active:
            return state + output
        parallel = (state.float() * output.float()).sum(-1, keepdim=True) / (
            state.float().square().sum(-1, keepdim=True) + 1e-6
        )
        return state + output - parallel.to(state.dtype) * state

    def extra_flops_per_token(self) -> int:
        return 12 * self.hidden_size if self.active else 0


class HighwayConnection(Connection):
    """Per-channel write gate ``2 * sigmoid(up(silu(down(x))))`` computed from the sublayer input, starting at 1."""

    def __init__(self, hidden_size: int, low_rank: int) -> None:
        super().__init__(hidden_size)
        self.down = nn.Linear(hidden_size, low_rank, bias=False)
        self.up = nn.Linear(low_rank, hidden_size, bias=False)

    def reset_parameters(self, generator: torch.Generator) -> None:
        local(self.down.weight).normal_(0.0, 0.02, generator=generator)
        local(self.up.weight).zero_()

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = rms_norm(state)
        return inputs, 2 * torch.sigmoid(self.up(F.silu(self.down(inputs))))

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        return state + context * output

    def extra_flops_per_token(self) -> int:
        return 6 * self.hidden_size


class RenormConnection(Connection):
    """Re-normalizes the stream after every write, times a learned per-connection scale."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.scale = nn.Parameter(torch.ones(1))

    def reset_parameters(self, generator: torch.Generator) -> None:
        local(self.scale).fill_(1.0)

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        return rms_norm(state + output) * self.scale.to(state.dtype)

    def extra_flops_per_token(self) -> int:
        return 6 * self.hidden_size


class DepthAttentionConnection(Connection):
    """Attention over depth: the sublayer reads the stream plus a per-token softmax mixture of the deltas of the last
    ``depth`` blocks, weighted by a query from the stream against keys from each delta (Attention Residuals, Kimi Team,
    2026, simplified). State: stream 0 is the residual stream, stream 1 the delta of the current block so far, streams
    2 onward the deltas of the previous ``depth`` blocks, newest first."""

    def __init__(self, hidden_size: int, sublayer: int, depth: int, dim: int) -> None:
        super().__init__(hidden_size)
        self.sublayer = sublayer
        self.depth = depth
        self.dim = dim
        self.streams = depth + 2
        self.query = nn.Linear(hidden_size, dim, bias=False)
        self.key = nn.Linear(hidden_size, dim, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))

    def initial_state(self, x0: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [x0, torch.zeros(*x0.shape[:-1], (self.streams - 1) * self.hidden_size, dtype=x0.dtype, device=x0.device)],
            dim=-1,
        )

    def enter(self, state, x0, resid_lambda, x0_lambda):
        streams = split_streams(state, self.streams, self.hidden_size)
        stream = resid_lambda.to(state.dtype) * streams[..., 0, :] + x0_lambda.to(state.dtype) * x0
        return torch.cat([stream.unsqueeze(-2), streams[..., 1:, :]], dim=-2).flatten(-2)

    def reset_parameters(self, generator: torch.Generator) -> None:
        local(self.query.weight).normal_(0.0, 0.02, generator=generator)
        local(self.key.weight).normal_(0.0, 0.02, generator=generator)
        local(self.beta).fill_(0.1)

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, None]:
        streams = split_streams(state, self.streams, self.hidden_size)
        stream, ring = streams[..., 0, :], streams[..., 2:, :]
        query = self.query(rms_norm(stream)).unsqueeze(-1)
        scores = self.key(rms_norm(ring)) @ query / self.dim**0.5
        mixture = (ring * F.softmax(scores, dim=-2)).sum(dim=-2)
        return rms_norm(stream + self.beta.to(state.dtype) * mixture), None

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        streams = split_streams(state, self.streams, self.hidden_size)
        stream = (streams[..., 0, :] + output).unsqueeze(-2)
        if self.sublayer == 0:
            return torch.cat([stream, output.unsqueeze(-2), streams[..., 2:, :]], dim=-2).flatten(-2)
        delta = (streams[..., 1, :] + output).unsqueeze(-2)
        pending = torch.zeros_like(streams[..., 1, :]).unsqueeze(-2)
        ring = torch.cat([delta, streams[..., 2:-1, :]], dim=-2)
        return torch.cat([stream, pending, ring], dim=-2).flatten(-2)

    def extra_flops_per_token(self) -> int:
        # The key projection runs once per ring entry; its parameters are only charged once by the 6-per-parameter count.
        return 6 * (self.depth - 1) * self.hidden_size * self.dim + 6 * self.depth * (self.dim + self.hidden_size)


class StaticStreamsConnection(Connection):
    """Gated residual without its matmuls: ``streams`` streams with per-stream RMS norm, static per-(stream, channel)
    read gains and static per-stream write scales. ``rho`` gives every stream a forget factor, applied once per block
    (fixed, the block's MLP write); None keeps pure accumulators."""

    def __init__(self, hidden_size: int, sublayer: int, streams: int, rho: list[float] | None = None) -> None:
        super().__init__(hidden_size)
        self.streams = streams
        self.sublayer = sublayer
        self.rho = rho
        self.norm_weight = nn.Parameter(torch.zeros(streams * hidden_size))
        self.read_gain = nn.Parameter(torch.zeros(streams * hidden_size))
        self.write_scale = nn.Parameter(torch.ones(streams))

    def reset_parameters(self, generator: torch.Generator) -> None:
        local(self.norm_weight).zero_()
        local(self.read_gain).zero_()
        local(self.write_scale).normal_(1.0, 0.05, generator=generator)

    def _read_streams(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        streams = split_streams(state, self.streams, self.hidden_size)
        normalized = rms_norm(streams) * (1.0 + self.norm_weight.to(state.dtype).view(self.streams, self.hidden_size))
        return streams, normalized

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, None]:
        _, normalized = self._read_streams(state)
        gains = 1.0 + self.read_gain.to(state.dtype).view(self.streams, self.hidden_size)
        return (gains * normalized).mean(dim=-2), None

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        streams = split_streams(state, self.streams, self.hidden_size)
        if self.rho is not None and self.sublayer == 1:
            streams = streams * torch.tensor(self.rho, dtype=state.dtype, device=state.device).unsqueeze(-1)
        scales = self.write_scale.to(state.dtype).unsqueeze(-1)
        return (streams + scales * output.unsqueeze(-2)).flatten(-2)


class StaticStreamsFinal(StaticStreamsConnection):
    def __init__(self, hidden_size: int, streams: int, rho: list[float] | None = None) -> None:
        super().__init__(hidden_size, sublayer=1, streams=streams, rho=rho)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.read(state, positions=None)[0]


class RecycledGateConnection(Connection):
    """Design RHG: gated residual whose gates are computed from the previous block's MLP hidden activation (width
    ``streams * hidden_size``, already paid for) instead of low-rank matmuls. The state carries the residual streams
    followed by the last hidden activation. Gate = 2*sigmoid(a * hidden + b) per (stream, channel), write scale =
    2*sigmoid(alpha * mean(hidden) + beta) per stream."""

    def __init__(self, hidden_size: int, sublayer: int, streams: int) -> None:
        super().__init__(hidden_size)
        assert streams == 4, "The MLP hidden activation has width 4 * hidden_size"
        self.sublayer = sublayer
        self.num_streams = streams
        self.streams = 2 * streams
        expanded = streams * hidden_size
        self.norm_weight = nn.Parameter(torch.zeros(expanded))
        self.gate_a = nn.Parameter(torch.zeros(expanded))
        self.gate_b = nn.Parameter(torch.zeros(expanded))
        self.write_alpha = nn.Parameter(torch.zeros(streams))
        self.write_beta = nn.Parameter(torch.zeros(streams))

    def reset_parameters(self, generator: torch.Generator) -> None:
        for parameter in (self.norm_weight, self.gate_a, self.gate_b, self.write_alpha):
            local(parameter).zero_()
        local(self.write_beta).normal_(0.0, 0.1, generator=generator)

    def initial_state(self, x0: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(*x0.shape[:-1], self.num_streams * self.hidden_size, dtype=x0.dtype, device=x0.device)
        return torch.cat([x0.repeat(1, 1, self.num_streams), zeros], dim=-1)

    def enter(self, state, x0, resid_lambda, x0_lambda):
        streams = split_streams(state, self.streams, self.hidden_size)
        n = self.num_streams
        residual = resid_lambda.to(state.dtype) * streams[..., :n, :] + x0_lambda.to(state.dtype) * x0.unsqueeze(-2)
        return torch.cat([residual, streams[..., n:, :]], dim=-2).flatten(-2)

    def _gates(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        streams = split_streams(state, self.streams, self.hidden_size)
        n = self.num_streams
        residual, hidden = streams[..., :n, :], streams[..., n:, :]
        normalized = rms_norm(residual) * (1.0 + self.norm_weight.to(state.dtype).view(n, self.hidden_size))
        scale = torch.rsqrt(hidden.float().square().mean(dim=(-2, -1), keepdim=True) + 1e-6).to(state.dtype)
        hidden_hat = hidden * scale
        return residual, normalized, hidden_hat

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, normalized, hidden_hat = self._gates(state)
        n = self.num_streams
        a = self.gate_a.to(state.dtype).view(n, self.hidden_size)
        b = self.gate_b.to(state.dtype).view(n, self.hidden_size)
        gates = 2 * torch.sigmoid(a * hidden_hat + b)
        write_scale = 2 * torch.sigmoid(
            self.write_alpha.to(state.dtype) * hidden_hat.mean(dim=-1) + self.write_beta.to(state.dtype)
        )
        return (gates * normalized).mean(dim=-2), write_scale

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        streams = split_streams(state, self.streams, self.hidden_size)
        n = self.num_streams
        residual = streams[..., :n, :] + context.unsqueeze(-1) * output.unsqueeze(-2)
        recycled = streams[..., n:, :] if self.sublayer == 0 else hidden.unflatten(-1, (n, self.hidden_size))
        return torch.cat([residual, recycled], dim=-2).flatten(-2)


class RecycledGateFinal(RecycledGateConnection):
    def __init__(self, hidden_size: int, streams: int) -> None:
        super().__init__(hidden_size, sublayer=1, streams=streams)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.read(state, positions=None)[0]


class DepthReplayConnection(Connection):
    """Design CDR: the plain stream, plus a read-time, zero-sum, per-channel replay of the last ``window`` sublayer
    outputs (L2-normalized), weighted by a per-channel softmax over depth with static scores and a magnitude term.
    Starts exactly plain (zero weights, uniform softmax gives zero-sum weight zero)."""

    def __init__(self, hidden_size: int, sublayer: int, window: int) -> None:
        super().__init__(hidden_size)
        self.sublayer = sublayer
        self.window = window
        self.streams = window + 1
        self.theta = nn.Parameter(torch.zeros(window * hidden_size))
        self.beta = nn.Parameter(torch.zeros(window * hidden_size))
        self.tau = nn.Parameter(torch.ones(1))

    def reset_parameters(self, generator: torch.Generator) -> None:
        local(self.theta).zero_()
        local(self.beta).zero_()
        local(self.tau).fill_(1.0)

    def initial_state(self, x0: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(*x0.shape[:-1], self.window * self.hidden_size, dtype=x0.dtype, device=x0.device)
        return torch.cat([x0, zeros], dim=-1)

    def enter(self, state, x0, resid_lambda, x0_lambda):
        streams = split_streams(state, self.streams, self.hidden_size)
        stream = resid_lambda.to(state.dtype) * streams[..., 0, :] + x0_lambda.to(state.dtype) * x0
        return torch.cat([stream.unsqueeze(-2), streams[..., 1:, :]], dim=-2).flatten(-2)

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, None]:
        streams = split_streams(state, self.streams, self.hidden_size)
        stream, ring = streams[..., 0, :], streams[..., 1:, :]
        theta = self.theta.to(state.dtype).view(self.window, self.hidden_size)
        beta = self.beta.to(state.dtype).view(self.window, self.hidden_size)
        probabilities = F.softmax(theta + beta * ring.abs(), dim=-2)
        weights = self.tau.to(state.dtype) * self.window * (probabilities - 1 / self.window)
        return rms_norm(stream + (weights * ring).sum(dim=-2)), None

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        streams = split_streams(state, self.streams, self.hidden_size)
        stream = (streams[..., 0, :] + output).unsqueeze(-2)
        ring = torch.cat([rms_norm(output).unsqueeze(-2), streams[..., 1:-1, :]], dim=-2)
        return torch.cat([stream, ring], dim=-2).flatten(-2)


class CausalFirConnection(Connection):
    """Design FIR: the sublayer reads the stream plus a learned per-channel causal FIR over the previous ``taps`` tokens
    of the same row (sequence-axis residual; rows can span documents, as attention does), zero-initialized so it starts plain."""

    def __init__(self, hidden_size: int, taps: int) -> None:
        super().__init__(hidden_size)
        self.taps = taps
        self.coefficients = nn.Parameter(torch.zeros(taps * hidden_size))

    def reset_parameters(self, generator: torch.Generator) -> None:
        local(self.coefficients).zero_()

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, None]:
        taps = self.coefficients.to(state.dtype).view(self.taps, self.hidden_size)
        filtered = state
        for k in range(1, self.taps + 1):
            valid = (positions >= k).unsqueeze(-1).to(state.dtype)
            filtered = filtered + taps[k - 1] * torch.roll(state, k, dims=1) * valid
        return rms_norm(filtered), None


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Doubly stochastic matrix from ``exp(logits)`` by alternate row/column normalization (in log space)."""
    log_p = logits.float()
    for _ in range(iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


class HyperConnection(Connection):
    """Hyper-connections (Zhu et al., 2024, arXiv:2409.19606): ``streams`` residual streams H [n, d]. The sublayer
    reads ``rms_norm(sum_s a_s H_s)`` and the streams become ``A^T H + b (x) output`` with a learned read vector a,
    stream-mixing matrix A [n, n] and write vector b. Static values start exactly plain (a = one-hot of the
    sublayer's index mod n, A = I, b = 1); ``dynamic`` adds ``scale * tanh(rms_norm(H_s) W)`` token-wise
    corrections with zero-initialized W (the paper's DHC).

    ``mix``: "free" (the paper), "sinkhorn" (mHC-style: A is the doubly stochastic Sinkhorn projection of
    exp(logits), a and b are positive via 2*sigmoid), or "none" (A = I fixed: static read/write only).
    ``read_streams`` "onehot" (paper init) or "mean" (a = 1/n)."""

    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        sublayer: int,
        streams: int,
        dynamic: bool = False,
        mix: str = "free",
        dynamic_scale: float = 0.01,
        read_init: str = "onehot",
        sinkhorn_iters: int = 10,
        sinkhorn_temp: float = 4.0,
    ) -> None:
        super().__init__(hidden_size)
        self.streams = streams
        self.index = (2 * layer_idx + sublayer) % streams
        self.dynamic = dynamic
        self.mix = mix
        self.dynamic_scale = dynamic_scale
        self.read_init = read_init
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temp = sinkhorn_temp
        # 1-D parameters (the extra-scalar optimizer group): read vector, flattened mixing matrix, write vector.
        self.read_weight = nn.Parameter(torch.zeros(streams))
        self.mix_weight = nn.Parameter(torch.zeros(streams * streams))
        self.write_weight = nn.Parameter(torch.zeros(streams))
        if dynamic:
            # token-wise corrections from each normalized stream: [read, write, mix row (n)]
            self.dynamic_proj = nn.Linear(hidden_size, streams + 2, bias=False)
        if mix == "none":
            self.mix_weight.requires_grad_(False)

    def reset_parameters(self, generator: torch.Generator) -> None:
        n = self.streams
        read = torch.zeros(n)
        if self.read_init == "mean":
            read.fill_(1.0 / n)
        else:
            read[self.index] = 1.0
        eye = torch.eye(n)
        if self.mix == "sinkhorn":
            # 2*sigmoid(0) = 1 for the write; read logits so that 2*sigmoid gives the one-hot-ish start
            local(self.read_weight).copy_(torch.where(read > 0, torch.zeros(n), torch.full((n,), -12.0)))
            local(self.mix_weight).copy_((eye * self.sinkhorn_temp).flatten())
            local(self.write_weight).zero_()
        else:
            local(self.read_weight).copy_(read)
            local(self.mix_weight).copy_(eye.flatten())
            local(self.write_weight).fill_(1.0)
        if self.dynamic:
            local(self.dynamic_proj.weight).zero_()

    def _coefficients(self, state: torch.Tensor):
        n = self.streams
        read = self.read_weight.float()
        write = self.write_weight.float()
        mix = self.mix_weight.float().view(n, n)
        if self.mix == "sinkhorn":
            read = 2 * torch.sigmoid(read)
            write = 2 * torch.sigmoid(write)
            mix = sinkhorn(mix, self.sinkhorn_iters)
        elif self.mix == "none":
            mix = torch.eye(n, device=state.device, dtype=torch.float32)
        read = read.to(state.dtype)
        write = write.to(state.dtype)
        mix = mix.to(state.dtype)
        if not self.dynamic:
            return read, mix, write
        streams = split_streams(state, n, self.hidden_size)
        delta = torch.tanh(self.dynamic_proj(rms_norm(streams))) * self.dynamic_scale  # [..., n, n + 2]
        read = read + delta[..., 0]  # [..., n]
        write = write + delta[..., 1]
        mix = mix + delta[..., 2:]  # [..., n (from), n (to)]
        return read, mix, write

    def read(self, state: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        read, mix, write = self._coefficients(state)
        streams = split_streams(state, self.streams, self.hidden_size)
        inputs = (read.unsqueeze(-1) * streams).sum(dim=-2)
        return rms_norm(inputs), (mix, write)

    def write(self, state, output, context, hidden=None) -> torch.Tensor:
        mix, write = context
        streams = split_streams(state, self.streams, self.hidden_size)
        if mix.dim() == 2:
            mixed = torch.einsum("st,...sd->...td", mix, streams)
        else:
            mixed = torch.einsum("...st,...sd->...td", mix, streams)
        return (mixed + write.unsqueeze(-1) * output.unsqueeze(-2)).flatten(-2)

    def extra_flops_per_token(self) -> int:
        n, d = self.streams, self.hidden_size
        flops = 6 * (n * n + 2 * n) * d  # read sum, stream mixing, write
        if self.dynamic:
            flops += 6 * (n - 1) * d * (n + 2) + 12 * n * d  # projection applied to every stream, stream norms
        return flops


class HyperFinal(nn.Module):
    """Sum of the streams, RMS-normalized (Zhu et al.)."""

    def __init__(self, hidden_size: int, streams: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.streams = streams

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return rms_norm(split_streams(state, self.streams, self.hidden_size).sum(dim=-2))


class RMSNormFinal(nn.Module):
    """Normalizes the first stream."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return rms_norm(state[..., : self.hidden_size])


class GatedFinal(GatedConnection):
    """The gated residual mixer collapsing the streams before the LM head, without injection."""

    def __init__(self, hidden_size: int, streams: int, low_rank: int, read_scale: float = 1.0) -> None:
        super().__init__(hidden_size, streams, low_rank, injection=False, read_scale=read_scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.read(state, positions=None)[0]


def make_connection(config, layer_idx: int, sublayer: int) -> Connection:
    residual = dict(config.residual)
    kind = residual.pop("type")
    hidden_size = config.hidden_size
    match kind:
        case "plain":
            return PlainConnection(hidden_size)
        case "gated":
            return GatedConnection(
                hidden_size, residual["streams"], residual["low_rank"], read_scale=residual.get("read_scale", 1.0)
            )
        case "static_streams":
            return StaticStreamsConnection(hidden_size, sublayer, residual["streams"], residual.get("rho"))
        case "recycled_gate":
            return RecycledGateConnection(hidden_size, sublayer, residual["streams"])
        case "depth_replay":
            return DepthReplayConnection(hidden_size, sublayer, residual["window"])
        case "causal_fir":
            return CausalFirConnection(hidden_size, residual["taps"])
        case "orthogonal":
            return OrthogonalConnection(hidden_size, sublayer, residual.get("sublayers", "both"))
        case "highway":
            return HighwayConnection(hidden_size, residual["low_rank"])
        case "renorm":
            return RenormConnection(hidden_size)
        case "depth_attention":
            return DepthAttentionConnection(hidden_size, sublayer, residual["depth"], residual["dim"])
        case "hyper":
            return HyperConnection(
                hidden_size,
                layer_idx,
                sublayer,
                residual["streams"],
                dynamic=residual.get("dynamic", False),
                mix=residual.get("mix", "free"),
                dynamic_scale=residual.get("dynamic_scale", 0.01),
                read_init=residual.get("read_init", "onehot"),
                sinkhorn_iters=residual.get("sinkhorn_iters", 10),
                sinkhorn_temp=residual.get("sinkhorn_temp", 4.0),
            )
        case _:
            raise ValueError(f"Unknown residual connection {kind!r}")


def make_final_norm(config) -> nn.Module:
    residual = config.residual
    match residual["type"]:
        case "gated":
            return GatedFinal(
                config.hidden_size,
                residual["streams"],
                residual["low_rank"],
                read_scale=residual.get("read_scale", 1.0),
            )
        case "static_streams":
            return StaticStreamsFinal(config.hidden_size, residual["streams"], residual.get("rho"))
        case "recycled_gate":
            return RecycledGateFinal(config.hidden_size, residual["streams"])
        case "hyper":
            return HyperFinal(config.hidden_size, residual["streams"])
        case _:
            return RMSNormFinal(config.hidden_size)
